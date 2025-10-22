import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st

load_dotenv()
os.getenv("GROQ_API_KEY")

model = os.getenv("MODEL")

llm = ChatGroq(
    model=model
)

# app config
st.set_page_config(page_title="QSS bot")
st.title("QSS bot")

# Load all PDF files from the data directory
data_dir = "data"

if not os.path.exists(data_dir):
    print(f"Error: Data directory not found at {data_dir}")
    exit()

# Get all PDF files in the data directory
pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]

if not pdf_files:
    print(f"Error: No PDF files found in {data_dir}")
    exit()

# print(f"Loading {len(pdf_files)} PDF file(s): {', '.join(pdf_files)}")

# Load all PDFs
documents = []
for pdf_file in pdf_files:
    pdf_path = os.path.join(data_dir, pdf_file)
    loader = PyPDFLoader(pdf_path)  # reads a PDF file and converts it into text pages.
    documents.extend(loader.load())  # list of text blocks (one per page)

# print(f"Loaded {len(documents)} pages from all PDFs")

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
texts = splitter.split_documents(documents)

# Create embeddings and vectorStore
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorStore = FAISS.from_documents(texts, embeddings)
retriever = vectorStore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


def get_response(user_query, chat_history):
    # Retrieve relevant documents
    relevant_docs = retriever.invoke(user_query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Format chat history for better readability
    formatted_history = ""
    for message in chat_history:
        if isinstance(message, HumanMessage):
            formatted_history += f"Human: {message.content}\n"
        elif isinstance(message, AIMessage):
            formatted_history += f"Assistant: {message.content}\n"

    # Create prompt template
    template = """
You are a helpful assistant for question-answering tasks about QSS policies and documents.

You have access to two sources of information:
1. Retrieved context from QSS policy documents
2. Previous conversation history

Instructions:
- First, check if the question can be answered from the chat history (e.g., follow-up questions, clarifications about previous answers)
- If the answer is in the chat history, use that information
- If not, use the retrieved context from the documents to answer
- If the answer is not available in either source, say that you don't have that information
- Keep the answer concise and accurate

Retrieved Context from Documents:
{context}

Conversation History:
{chat_history}

Current Question: {user_question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "context": context,
        "chat_history": formatted_history,
        "user_question": user_query,
    })


# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am QSS bot. How can I help you?"),
    ]


# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)


# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    # Add user message to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    # Display user message immediately
    with st.chat_message("Human"):
        st.write(user_query)

    # Generate and display AI response with streaming
    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    # Add AI response to chat history
    st.session_state.chat_history.append(AIMessage(content=response))
