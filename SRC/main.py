from langchain_community.llms import GooglePalm
from embedding import embedding_model
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

load_dotenv()

token = os.getenv("Google_api_key")

from langchain_google_genai import GoogleGenerativeAI
llm_google_palm = GoogleGenerativeAI(model="gemini-pro",google_api_key=token, temperature=0.1, max_tokens= 100)

#llm_google_palm = GooglePalm(google_api_key=token, temperature=0.1, max_tokens= 100)

system_prompt = "<s>[INST] You are an expert mental health counseling chatbot named Mindguardian. You provide professional mental health counseling to users. [/INST]"
conversational_memory_length = 10
memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)
print("System prompt and memory initialized.")

def query_chromdb(user_query):
    # Generate the query vector from the user's input
    query_vector = embedding_model.encode(user_query).tolist()
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=1,
        include=["metadatas"]  
    )
    return response

# Example usage
response = query_chromdb("I've been feeling a bit off. I sometimes find it hard to focus and concentrate on tasks, but it doesn't happen too often. I have been feeling a bit disconnected from my friends, but we still hang out and talk normally. My sleeping habits haven't changed much")
print(response['metadatas'])

def query_llm(user_question, _):
    try:
        print(f"Received user question: {user_question}")
        context = query_chromdb(user_question)

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=system_prompt
                ),
                MessagesPlaceholder(
                    variable_name="chat_history"
                ),
                SystemMessage(
                    content=f"<s>[INST] Use this context only if relevant to user query: {context} [/INST]"
                ),
                HumanMessagePromptTemplate.from_template(
                    "<s>[INST] User query: {human_input} [/INST]"
                ),
            ]
        )
        print("Prompt constructed.")

        conversation = LLMChain(
            llm=llm_google_palm, 
            prompt=prompt,
            verbose=False,
            memory=memory,
        )
        print("LLMChain initialized.")
        response = conversation.predict(human_input=user_question)
        print(f"LLM response: {response}")

        return response
    except Exception as e:
        print(f"Error in query_llm: {e}")
        return "Sorry, something went wrong. Please try again."

# Default message for the chatbot
default_message = """I'm MindGuardian, a mental health counseling chatbot. How can I help you?"""

gradio_interface = gr.ChatInterface(
    query_llm,
    chatbot=gr.Chatbot(value=[[None, default_message]]),
    textbox=gr.Textbox(placeholder="Type your query", container=False, scale=7),
    title="Mindguardian, a mental health counseling chatbot",
    theme='gradio/base',
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
)

# Launch the interface
print("Launching Gradio interface...")
gradio_interface.launch(debug=True)    