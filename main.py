import os
import re
import shutil
import hashlib
import tempfile
from typing import TypedDict
from datetime import timedelta
from langchain.chains import RetrievalQA
from werkzeug.utils import secure_filename
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from flask import Flask, render_template, request, session
from langchain_core.messages import AIMessage, HumanMessage 
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#####################################################################################
app = Flask(__name__)
app.secret_key = 'king123'
app.permanent_session_lifetime = timedelta(minutes=30)  

ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 10*1024*1024 #10 MB

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

AVAILABLE_MODELS = {
    'qwen-0.5b': {
        'path': 'models/qwen2.5-0.5b-instruct-q8_0.gguf',
        'name': 'Qwen 0.5B',
        'n_ctx': 2048,
        'description': 'Fast and efficient'
    },
    'qwen-1.5b': {
        'path': 'models/Qwen2.5-1.5B-Instruct-iq3_s.gguf',
        'name': 'Qwen 1.5B',
        'n_ctx': 4096,
        'description': 'Better quality responses'
    },
    'phi2': {
        'path': 'models/phi-2.Q3_K_M.gguf',
        'name': 'Phi-2',
        'n_ctx': 4096,
        'description': 'Balanced performance'
    },
    'tinyllama': {
        'path': 'models/tinyllama-1.1b-chat-v1.0.Q3_K_L.gguf',
        'name': 'TinyLlama',
        'n_ctx': 2048,
        'description': 'Ultra fast'
    }
}

model_cache = {}
def get_llm(model_key='qwen-0.5b'):
	if model_key not in model_cache:
		model_info = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS['qwen-0.5b'])

		if not os.path.exists(model_info['path']):
			print(f"Model {model_info['path']} not found, falling back to default")
			model_key='qwen-0.5b'
			model_info = AVAILABLE_MODELS[model_key]
		model_cache[model_key] = LlamaCpp(
			model_path=model_info['path'],
			n_ctx=model_info['n_ctx'],
			temperature=0.3,
			verbose=False,
			n_threads=4, 
			n_batch=256,
			use_mmap=True,
			use_mlock=True)
	return model_cache[model_key]

embeddings = OllamaEmbeddings(model='all-minilm')
messages = {}
vector_cache = {}  
def process_pdf(path, file_hash):
	try:
	    loader = PyPDFLoader(path)
	    pages = loader.load()
	    if not pages:
	    	raise ValueError('No content extracted from PDF')
	    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
	    docs = splitter.split_documents(pages)
	    vectorstore = FAISS.from_documents(docs, embeddings)
	    os.makedirs('faiss_indexes', exist_ok=True)
	    vectorstore.save_local(f"faiss_indexes/{file_hash}")
	except Exception as e:
		print(f"Error processing PDF: {e}")
		raise

def hist(id):
	if id not in messages:
		messages[id] = ChatMessageHistory()
	return messages[id]


prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


class AState(TypedDict):
	input: str
	output: str 
	session_id: str 
	file_hash: str 
	selected_model: str 

def router_fn(state):

	selected_model = state.get('selected_model', 'qwen-0.5b')
	llm = get_llm(selected_model)
	doc_keywords = ['document', 'pdf', 'file', 'according to', 'in the document', 'what does it say', 'summarize']
	user_input = state['input'].lower()
	if any(keyword in user_input for keyword in doc_keywords):
		return 'Doc'

	prompt = f'''
	Classify this as "Doc" if asking about uploaded document content, "Chat" for general conversation:
	"{state['input']}"
	Reply with exactly one word: Doc or Chat'''
	try:
		result = llm.invoke(prompt).strip().lower()
		return 'Doc' if 'doc' in result.lower() else 'Chat'
	except:
		return 'Chat'
	

def chat_node(state):
	selected_model = state.get('selected_model', 'qwen-0.5b')
	llm = get_llm(selected_model)

	session_id = state['session_id']

	core_chain = prompt | llm 
	chat_chain = RunnableWithMessageHistory(
		runnable=core_chain,
		get_session_history=hist,
		input_messages_key='input',
		history_messages_key='history')

	result = chat_chain.invoke(
		{'input':state['input']},
		config={'configurable':{'session_id':session_id}})
	return {'output':result, 'session_id':session_id}

def doc_node(state):
	selected_model = state.get('selected_model', 'qwen-0.5b')
	llm = get_llm(selected_model)

	file_hash = state.get('file_hash')
	user_input = state.get('input')
	if not file_hash:
		return {'output': "No document uploaded.", 'session_id': state.get('session_id')}
	
	vectorstore = vector_cache.get(file_hash)
	if not vectorstore:
		try:
			vectorstore = FAISS.load_local(f'faiss_indexes/{file_hash}', embeddings, allow_dangerous_deserialization=True)
			vector_cache[file_hash] = vectorstore
		except Exception as e:
			return {'output': f"Error loading document: {str(e)}", 'session_id': state.get('session_id')}

	retriever = vectorstore.as_retriever()
	qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
	result = qa.run(user_input)
	
	return {'output': result, 'session_id': state.get('session_id')}

def build_graph():
	builder = StateGraph(AState)
	builder.add_node('Chat', chat_node)
	builder.add_node('Doc', doc_node)
	builder.add_node('Router', lambda x: x)
	builder.add_conditional_edges('Router', router_fn, {
		'Chat':'Chat',
		'Doc': 'Doc'
		})
	builder.set_entry_point('Router')
	builder.add_edge('Chat', END)
	builder.add_edge('Doc', END)
	graph = builder.compile()
	return graph

graph = build_graph()

def chain_(file_hash, user_msg, selected_model='qwen-0.5b'):
	session_id = session.get('session_id')
	if not session_id:
		session_id = hashlib.sha256(os.urandom(16)).hexdigest()
		session['session_id'] = session_id
	state = {
	'input': user_msg,
	'session_id': session_id,
	'file_hash': file_hash,
	'selected_model': selected_model
	}
	reply = graph.invoke(state)
	return reply['output']

@app.route('/', methods=['GET'])
def index():
	session.clear()
	return render_template('file.html')


@app.route('/q', methods=['POST', 'GET'])
def aa():
	session.permanent = True
	if 'file_message' not  in session:
		session['file_message'] = ''

	if 'chat_history' not in session:
		session['chat_history'] = []

	if 'selected_model' not in session:
		session['selected_model'] = 'qwen-0.5b'

	if request.method == 'POST':
		#handle model selection
		selected_model = request.form.get('selected_model', 'qwen-0.5b')
		if selected_model != session.get('selected_model'):
			session['selected_model'] = selected_model
			session['file_message'] = f"Switched to {AVAILABLE_MODELS[selected_model]['name']}"

		# Handle file upload
		uploaded_file = request.files.get('file')
		# print(uploaded_file)
		if uploaded_file and uploaded_file.filename != '':
			session['chat_history'] = []

			if not allowed_file(uploaded_file.filename):
				session['file_message'] = 'Only PDF files are allowed.'
				return render_template('q.html', messages=session.get('chat_history', []), mes=session['file_message'])
			
			#check file size
			uploaded_file.seek(0, os.SEEK_END)
			file_size = uploaded_file.tell()
			uploaded_file.seek(0)

			if file_size > MAX_FILE_SIZE:
				session['file_message'] = "File too large. Maximum size is 10MB."
				return render_template("q.html", messages=session.get('chat_history', []), mes=session['file_message'])

			temp_dir = tempfile.mkdtemp()
			file_path = os.path.join(temp_dir, uploaded_file.filename)
			uploaded_file.save(file_path)
			with open(file_path, 'rb') as f:
				file_hash = hashlib.sha256(f.read()).hexdigest()
			session['file_message'] = f"'{uploaded_file.filename}' uploaded successfully."
			session['file_hash'] = file_hash
			faiss_dir = os.path.join('faiss_indexes', file_hash)
			if not os.path.exists(faiss_dir):
				process_pdf(file_path, file_hash)
			shutil.rmtree(temp_dir)
		# Handle message input
		user_msg = request.form.get('message', '').strip()
		if user_msg:
			file_hash = session.get('file_hash')
			if not file_hash:
				return "No file was uploaded", 400
			session['chat_history'].append({"role": "user", "text": user_msg})
			reply = chain_(file_hash, user_msg)  
			session['chat_history'].append({"role": "bot", "text": reply})
	return render_template("q.html", 
    	messages=session.get('chat_history', []),
    	mes=session.get('file_message', ''),
    	selected_model=session.get('selected_model', 'qwen-0.5b'))
