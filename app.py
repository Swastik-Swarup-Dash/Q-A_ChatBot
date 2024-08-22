import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Set environment variables for LangChain and Google API keys
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A ChatBot With Gemini"
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to user queries."),
        ("user", "Question: {question}")
    ]
)

# Function to generate response
def generate_response(question, model_name, temperature, max_tokens):
    # Initialize the Google Generative AI model with correct parameters
    llm = GoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Create output parser
    output_parser = StrOutputParser()
    
    # Combine prompt, model, and parser
    chain = prompt | llm | output_parser
    
    # Generate response
    answer = chain.invoke({"question": question})
    return answer

# Streamlit UI components
st.title("Q&A ChatBot With Gemini")
st.sidebar.title("Settings")

# Model selection
llm_selection = st.sidebar.selectbox("Select a Gemini Model", ["Gemini 1.0 Pro", "Gemini 1.5 Flash"])

# Map the selection to the actual model name used by the API
model_name_mapping = {
    "Gemini 1.0 Pro": "models/gemini-1.0-pro",
    "Gemini 1.5 Flash": "models/gemini-1.5-flash"
}
model_name = model_name_mapping[llm_selection]

# Temperature and max tokens settings
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max tokens", min_value=50, max_value=500, value=150)

st.write("Go ahead and ask any question")

# Input and response handling
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, model_name, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide a query")

     






