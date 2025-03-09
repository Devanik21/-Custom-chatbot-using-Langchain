# app.py
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app_logger = logging.getLogger(__name__)

application = Flask(__name__)
api_interface = Api(application)

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("API key for Google services not found in environment variables. Ensure GOOGLE_API_KEY is set in your .env file.")

doc_vectorstore = None
chat_conversation_chain = None

def chatbot_setup():
    global doc_vectorstore, chat_conversation_chain

    try:
        app_logger.info("Fetching course data from BrainLox website...")
        web_loader = WebBaseLoader("https://brainlox.com/courses/category/technical")
        documents_data = web_loader.load()
        app_logger.info("Data loaded successfully from the website.")

        text_chunker = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150
        )
        document_chunks = text_chunker.split_documents(documents_data)
        app_logger.info(f"Documents split into {len(document_chunks)} text chunks.")

        embedding_model = GoogleGenerativeAIEmbeddings(
            google_api_key=google_api_key,
            model="models/embedding-001"
        )
        doc_vectorstore = FAISS.from_documents(document_chunks, embedding_model)
        app_logger.info("Embeddings created and stored in FAISS vector store.")

        llm_model = ChatGoogleGenerativeAI(
            google_api_key=google_api_key,
            model="gemini-1.5-flash",
            temperature=0.3
        )

        chat_conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm_model,
            retriever=doc_vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        app_logger.info("Chatbot initialization completed successfully.")
        return True

    except Exception as err:
        app_logger.error(f"Chatbot initialization encountered an error: {str(err)}")
        return False

chatbot_setup()


class BrainLoxChatbotResource(Resource):
    def post(self):
        global chat_conversation_chain

        if not chat_conversation_chain:
            initialization_success = chatbot_setup()
            if not initialization_success:
                return {"error": "Chatbot service is unavailable due to initialization failure."}, 500

        try:
            request_data = request.get_json()

            if not request_data:
                return {"error": "No input data was provided in the request."}, 400

            user_query = request_data.get("query")
            dialogue_history = request_data.get("chat_history", [])

            if not user_query:
                return {"error": "The 'query' parameter is missing from the request data."}, 400

            response_result = chat_conversation_chain(
                {"question": user_query, "chat_history": dialogue_history}
            )

            formatted_response = {
                "answer": response_result["answer"],
                "sources": [source_doc.metadata.get("source", "Unknown Source") for source_doc in response_result.get("source_documents", [])]
            }

            return formatted_response, 200

        except Exception as err:
            app_logger.error(f"Error encountered while processing chat request: {str(err)}")
            return {"error": "An error occurred while processing your chat request.", "details": str(err)}, 500

api_interface.add_resource(BrainLoxChatbotResource, "/api/chat")

@application.route("/health", methods=["GET"])
def application_health_check():
    global doc_vectorstore, chat_conversation_chain

    if doc_vectorstore and chat_conversation_chain:
        return jsonify({"status": "operational"}), 200
    else:
        return jsonify({"status": "degraded", "reason": "Chatbot service is not fully initialized."}), 503


if __name__ == "__main__":
    application.run(debug=True)