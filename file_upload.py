from flask import Flask, render_template, request, session
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp 
import os

app = Flask(__name__)
app.secret_key= 'hello123'

chat_history = []
vector_cache = {}
llm = LlamaCpp(
    model_path="models/tinyllama-1.1b-chat-v1.0.Q3_K_L.gguf",
    n_ctx=4096,
    temperature=0.5,
    verbose=True
)


def process_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    embeddings = OllamaEmbeddings(model='all-minilm')
    vectorstore = FAISS.from_documents(docs, embeddings)

    vector_cache[path] = vectorstore  # Cache it


def chain_(path, query):
    if path not in vector_cache:
        process_pdf(path)
    else:
        vectorstore = vector_cache[path]

  
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
    )

    response = qa_chain.invoke({"query": query})
    
    return response['result']
@app.route('/', methods=['GET'])
def index():
    return render_template('file.html')

@app.route('/q', methods=['POST', 'GET'])
def aa():

    if 'file_message' not  in session:
    	session['file_message'] = ''

    if 'file_path' not in session:
        session['file_path'] = ''

    if request.method == 'POST':
        # Handle file upload
        uploaded_file = request.files.get('file')
        # print(uploaded_file)
        if uploaded_file and uploaded_file.filename != '':
            # You can save it if needed:
            os.makedirs('uploads', exist_ok=True)
            file_loc = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(file_loc)
            session['file_message'] = f"'{uploaded_file.filename}' uploaded successfully."
            session['file_path'] = file_loc
            process_pdf(file_loc)
        # Handle message input
        user_msg = request.form.get('message', '').strip()
        if user_msg:
            file_loc = session.get('file_path')
            if not file_loc:
                return "No file was uploaded", 400

            chat_history.append({"role": "user", "text": user_msg})
            reply = chain_(file_loc, user_msg)  # Replace with your real response
            chat_history.append({"role": "bot", "text": reply})
            
    return render_template("q.html", messages=chat_history, mes=session['file_message'])



