import streamlit as st
from embedding import embedding_model
import chromadb
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAI
import logging
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize ChromaDB client and access the existing collection
directory = r"C:\Users\hsai5\OneDrive\Documents\LLM projects\conversational_RAG_chatbot\chroma_db"
chroma_client = chromadb.PersistentClient(path=directory)
collection = chroma_client.get_collection(name="mindguardian_collection")

# Initialize LLM
token = os.getenv("Google_api_key")
llm_gemini = GoogleGenerativeAI(model="gemini-pro", google_api_key=token, temperature=0.1, max_tokens=100)

# Set up system prompt and memory
system_prompt = "You are an expert mental health counseling chatbot named Mindguardian. You provide professional mental health counseling to users."
conversational_memory_length = 25
memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

def query_chromdb(user_query):
    query_vector = embedding_model.encode(user_query).tolist()
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=1,
        include=["metadatas"]  
    )
    logger.info(f"ChromaDB query: {user_query}")
    logger.info(f"ChromaDB response: {response}")
    return response

def query_llm(user_question):
    try:
        logger.info(f"Processing user question: {user_question}")
        context = query_chromdb(user_question)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            SystemMessage(content=f"<s>[INST] Use this context only if relevant to user query: {context} [/INST]"),
            HumanMessagePromptTemplate.from_template("<s>[INST] User query: {human_input} [/INST]"),
        ])

        conversation = LLMChain(
            llm=llm_gemini, 
            prompt=prompt,
            verbose=False,
            memory=memory,
        )
        response = conversation.predict(human_input=user_question)
        logger.info(f"Generated response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error in query_llm: {e}")
        return "Sorry, something went wrong. Please try again."

# Streamlit UI
st.set_page_config(page_title="Mindguardian", page_icon="🧠", layout="wide")
st.title("Mindguardian: Mental Health Counseling Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I'm MindGuardian, a mental health counseling chatbot. How can I help you?"}
    ]

if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What's on your mind?"):
    logger.info(f"User input: {prompt}")
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = query_llm(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a sidebar with additional information or controls
with st.sidebar:
    st.title("About Mindguardian")
    st.write("Mindguardian is an AI-powered mental health counseling chatbot designed to provide support and guidance.")
    st.write("Please note: This is not a substitute for professional medical advice, diagnosis, or treatment.")
    st.write(f"Current Chat ID: {st.session_state.chat_id}")

    if st.button("Clear Chat History"):
        logger.info("Chat history cleared")
        st.session_state.messages = [
            {"role": "assistant", "content": "I'm MindGuardian, a mental health counseling chatbot. How can I help you?"}
        ]
        st.session_state.chat_id = str(uuid.uuid4())
        st.experimental_rerun()

    if st.button("Download Chat History"):
        logger.info("Chat history downloaded")
        chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
        st.download_button(
            label="Download Chat",
            data=chat_history,
            file_name=f"chat_history_{st.session_state.chat_id}.txt",
            mime="text/plain"
        )