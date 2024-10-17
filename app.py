from waitress import serve
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from flask_compress import Compress

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = OpenAI(temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


# @app.route("/get", methods=["GET", "POST"])   BAPPY
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     response = rag_chain.invoke({"input": msg})
#     print("Response : ", response["answer"])
#     return str(response["answer"])


@app.route("/get", methods=["GET", "POST"])  #Aayush
def chat():
    msg = request.form["msg"]
    input = msg
    
    # Process input and get response
    response = rag_chain.invoke({"input": msg})
    
    # If response is large, consider truncating or compressing
    if len(response["answer"]) > 300:  # Adjust threshold as needed
        response["answer"] = response["answer"][:300] + "..."

    return str(response["answer"])


    
#     return str(response["answer"])

@app.route("/get_data", methods=["GET"])
def get_data():
    # Example of how to limit or paginate data
    page = request.args.get("page", 1, type=int)
    per_page = 100  # Limit to 100 results per page
    
    data = large_data_source.paginate(page, per_page)
    
    return jsonify(data)





# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port= 5000, debug= True)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))