# AskMyDoc

A Document Q&A application, where we can upload a document and ask questions
about it. It can also chat but the main purpose is for document Q&A

## USAGE
1. Before running make sure to first install the models specified in the code from hugginface hub and store it in the models folder.
2. You can also use your own desired models, but need to change the codes to mathc the models.
3. export FLASK_APP=main.py
4. flask run

### FEATURES
1. Has different models to choose from(Qwen, Hermes, Phi)
2. Can handle chatting and document Q&A
3. Used langgraph for routing between chat and Q&A
