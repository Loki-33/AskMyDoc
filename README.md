# AskMyDoc
A Local RAG Assistant
It only answers the document related questions for now, so chatting is not recommended

## USAGE:
1. create a models folder and download any model of gguf format from huggingface.co and put it in the models folder
2. make sure you have all the libraries
3. To use flask, type:
    export FLASK_APP=upload_file.py
    flask run
5. Then open localhost on your browser and test it.


## Future Implementations
1. Change from langchain to langGraph so that we can add a route to chatting and DOC Q&A
