import os
import tempfile
from pathlib import Path
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
from pathlib import Path
from typing import Dict, List

import streamlit as st

os.environ["USER_AGENT"] = "myagent"

from langchain import hub
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Load environment variables from .env file
load_dotenv()

# Set API keys from environment variables.
# In a real deployment, ensure these are properly set as environment variables
# in your hosting environment.
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Check if API keys are available
if (
    not os.environ["OPENAI_API_KEY"]
):
    st.error(
        "OPENAI_API_KEY not found. Please set it in your .env file or as an environment variable."
    )
    st.stop()
if (
    not os.environ["TAVILY_API_KEY"]
):
    st.error(
        "TAVILY_API_KEY not found. Please set it in your .env file or as an environment variable."
    )
    st.stop()


# Define a directory for persisting ChromaDB
PERSIST_DIRECTORY = "./chroma_db_uploaded"


# --- Graph State Definition ---
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]  # Changed to Document objects for richer info


# --- Graph Node Definitions ---
def retrieve(state, retriever_instance):
    """
    Retrieves documents based on the question using the provided retriever.
    """
    question = state["question"]
    documents = retriever_instance.get_relevant_documents(question)
    return {"documents": documents, "question": question}


def generate(state, rag_chain):
    """
    Generates an answer using the RAG chain.
    """
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def evaluate_documents(state, retrieval_grader):
    """
    Evaluates the retrieved documents for relevance.
    """
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"

    if not documents:
        # If no documents were retrieved at all, immediately go to web search
        web_search = "Yes"
    else:
        # Grade each document
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            if score.binary_score.lower() == "yes":  # Ensure case-insensitive check
                filtered_docs.append(d)

        # If less than 70% of documents are relevant, trigger web search
        # Avoid division by zero if original documents list is empty
        if len(documents) > 0 and (len(filtered_docs) / len(documents)) < 0.7:
            web_search = "Yes"
        elif (
            len(documents) == 0 and len(filtered_docs) == 0
        ):  # Case where documents were empty initially
            web_search = "Yes"

    return {
        "documents": filtered_docs,  # Only keep relevant documents for generation
        "question": question,
        "web_search": web_search,
    }


def transform_query(state, question_rewriter):
    """
    Rewrites the question for better web search.
    """
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def web_search_node(state, web_search_tool):
    """
    Performs a web search and adds results to documents.
    """
    question = state["question"]
    documents = state["documents"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results_doc = Document(
        page_content=web_results, metadata={"source": "web_search"}
    )
    documents.append(web_results_doc)  # Append web results as a new document
    return {"documents": documents, "question": question}


# --- Conditional Edge Logic ---
def decide_to_generate(state):
    """
    Decides whether to generate an answer or transform the query for web search.
    """
    return "transform_query" if state["web_search"] == "Yes" else "generate"


# --- 1. SETUP THE RAG-FUSION GRAPH ---
# Use Streamlit's caching to load the LLMs and graph structure only once.
@st.cache_resource
def setup_common_components():
    """
    Sets up LLMs, tools, and common chains/prompts that don't depend on the retriever.
    """
    rag_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    retrieval_evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    question_rewriter_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    web_search_tool = TavilySearchResults(k=3)

    # Chains and Prompts
    rag_prompt = hub.pull("rlm/rag-prompt")
    rag_chain = rag_prompt | rag_llm | StrOutputParser()

    # Retrieval Grader
    class RetrievalEvaluator(BaseModel):
        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    structured_llm_evaluator = retrieval_evaluator_llm.with_structured_output(
        RetrievalEvaluator
    )
    system_evaluator = """You are a document retrieval evaluator. If the document contains keywords or semantic meaning related to the question, grade it as relevant. 
    Output a binary score 'yes' or 'no'."""
    retrieval_evaluator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_evaluator),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )
    retrieval_grader = retrieval_evaluator_prompt | structured_llm_evaluator

    # Question Re-writer
    system_rewriter = """You are a question re-writer that converts an input question to a better version for web search. 
    Reason about the underlying semantic intent."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_rewriter),
            (
                "human",
                "Initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )
    question_rewriter = re_write_prompt | question_rewriter_llm | StrOutputParser()

    return {
        "rag_chain": rag_chain,
        "retrieval_grader": retrieval_grader,
        "question_rewriter": question_rewriter,
        "web_search_tool": web_search_tool,
    }


# Removed @st.cache_resource from compile_graph
def compile_graph(retriever_instance):
    """
    Compiles the LangGraph workflow, taking a retriever instance as input.
    This allows the retriever to be dynamic based on user choice/upload.
    """
    common_components = setup_common_components()

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", lambda state: retrieve(state, retriever_instance))
    workflow.add_node(
        "grade_documents",
        lambda state: evaluate_documents(state, common_components["retrieval_grader"]),
    )
    workflow.add_node(
        "generate", lambda state: generate(state, common_components["rag_chain"])
    )
    workflow.add_node(
        "transform_query",
        lambda state: transform_query(state, common_components["question_rewriter"]),
    )
    workflow.add_node(
        "web_search_node",
        lambda state: web_search_node(state, common_components["web_search_tool"]),
    )

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"},
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# --- Document Processing Functions ---
@st.cache_resource(hash_funcs={Path: lambda p: str(p)})
def get_persistent_chroma_retriever():
    """
    Attempts to load an existing Chroma vector store from the persistent directory.
    If it doesn't exist or is empty, it initializes a new (empty) one.
    This serves as the default retriever when no specific files are uploaded.
    """
    try:
        # Check if the persisted directory exists and contains data
        if Path(PERSIST_DIRECTORY).exists() and list(Path(PERSIST_DIRECTORY).iterdir()):
            with st.spinner("Loading existing database from persistent storage..."):
                vectorstore = Chroma(
                    persist_directory=PERSIST_DIRECTORY,
                    embedding_function=OpenAIEmbeddings(),
                )
                st.success("Loaded existing database from persistent storage!")
                return vectorstore.as_retriever()
        else:
            # If directory is empty or doesn't exist, create a new empty Chroma instance
            # This ensures the retriever always exists, even if it has no documents.
            with st.spinner("Initializing new (empty) persistent database..."):
                vectorstore = Chroma(
                    embedding_function=OpenAIEmbeddings(),
                    persist_directory=PERSIST_DIRECTORY,
                )
                vectorstore.persist()  # Persist the empty state
                st.info(
                    "No existing documents found. Agent will rely on web search if no files are uploaded."
                )
                return vectorstore.as_retriever()
    except Exception as e:
        st.warning(
            f"Could not load or initialize persistent database: {e}. Creating a new empty one."
        )
        # Fallback to creating a new empty one if an error occurs during loading
        vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(),
            persist_directory=PERSIST_DIRECTORY,
        )
        vectorstore.persist()
        return vectorstore.as_retriever()


@st.cache_resource(hash_funcs={Path: lambda p: str(p)})
def create_or_load_uploaded_retriever(uploaded_files_hashes):
    """
    Creates or loads a Chroma vector store from uploaded files and returns a retriever.
    If new files are provided (via uploaded_files_hashes), it rebuilds the DB by
    clearing existing content and adding new documents.
    Otherwise, it attempts to load the existing persistent DB.
    """
    is_new_upload = bool(uploaded_files_hashes)

    if is_new_upload:
        with st.spinner(
            "Processing uploaded files and updating database... This may take a moment."
        ):
            # Load the existing vectorstore or create a new one if it doesn't exist
            # We need to ensure a client is available to delete the collection
            try:
                # Initialize a Chroma client to interact with the persistent directory
                # This will load the existing DB if it exists, or create a new one
                vectorstore = Chroma(
                    persist_directory=PERSIST_DIRECTORY,
                    embedding_function=OpenAIEmbeddings(),
                )
            except Exception as e:
                st.warning(
                    f"Could not load existing database for update: {e}. Initializing a new one."
                )
                vectorstore = Chroma(
                    embedding_function=OpenAIEmbeddings(),
                    persist_directory=PERSIST_DIRECTORY,
                )

            # Delete all existing documents from the collection to "overwrite"
            try:
                vectorstore.delete_collection()
                vectorstore.persist()  # Persist the deletion
                vectorstore = Chroma(
                    embedding_function=OpenAIEmbeddings(),
                    persist_directory=PERSIST_DIRECTORY,
                )
                st.success("Existing database content cleared.")
            except Exception as e:
                # Catch any errors during the deletion process
                st.warning(
                    f"Could not clear existing documents from the database: {e}. Proceeding with adding new documents, which might lead to duplicates if not cleared."
                )
                # We don't stop here, but warn the user that old documents might remain.

            all_docs = []
            for uploaded_file in st.session_state["uploaded_files"]:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=uploaded_file.name
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                try:
                    if uploaded_file.type == "application/pdf":
                        loader = PyPDFLoader(tmp_file_path)
                    elif uploaded_file.type == "text/plain":
                        loader = TextLoader(tmp_file_path)
                    else:
                        st.warning(
                            f"Skipping unsupported file type: {uploaded_file.type}"
                        )
                        continue
                    docs = loader.load()
                    all_docs.extend(docs)
                finally:
                    os.unlink(tmp_file_path)

            if not all_docs:
                st.warning("No supported documents found in uploaded files to process.")
                return None

            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=250, chunk_overlap=0
            )
            doc_splits = text_splitter.split_documents(all_docs)

            # Add the new documents to the (now empty) collection
            vectorstore.add_documents(doc_splits)
            vectorstore.persist()
            st.session_state["uploaded_docs_processed"] = True
            st.success("Database updated and persisted with new uploaded files!")
            return vectorstore.as_retriever()
    else:  # No new files provided, try to load existing persistent DB
        try:
            if Path(PERSIST_DIRECTORY).exists() and list(
                Path(PERSIST_DIRECTORY).iterdir()
            ):
                with st.spinner("Loading existing database from uploaded files..."):
                    vectorstore = Chroma(
                        persist_directory=PERSIST_DIRECTORY,
                        embedding_function=OpenAIEmbeddings(),
                    )
                    st.success("Loaded existing database from uploaded files!")
                    return vectorstore.as_retriever()
            else:
                st.info(
                    "No uploaded files processed yet, or persistent database is empty."
                )
                return None  # Indicate no retriever from uploaded files is available
        except Exception as e:
            st.warning(
                f"Could not load existing database: {e}. Please re-upload files if needed."
            )
            return None  # Indicate failure to load


# --- 2. STREAMLIT UI ---
st.set_page_config(
    page_title="Adaptive RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("🤖 Adaptive RAG Agent")
st.markdown(
    """
This agent answers questions using the uploaded documents as the knowledge base. 
If the initial documents aren't relevant enough, it will automatically rewrite the question and search the web for more information.
"""
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = "web"  # Default to web knowledge base
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "uploaded_docs_processed" not in st.session_state:
    st.session_state.uploaded_docs_processed = False
if "current_retriever" not in st.session_state:
    st.session_state.current_retriever = None
if "app_graph" not in st.session_state:
    st.session_state.app_graph = None


# Sidebar for mode selection and file upload
with st.sidebar:
    st.header("Configuration")
    rag_mode_selection = st.radio(
        "Choose Knowledge Base:",
        (
            "Use existing persistent database (or rely on web search)",
            "Upload my own files",
        ),
        key="rag_mode_radio",
        on_change=lambda: st.session_state.update(
            rag_mode=st.session_state.rag_mode_radio,
            uploaded_docs_processed=False,  # Reset processing status on mode change
            current_retriever=None,
            app_graph=None,
            messages=[],  # Clear chat history on mode change
        ),
    )

    if rag_mode_selection == "Upload my own files":
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="file_uploader",
        )
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            if st.button("Process Uploaded Files and Create Database"):
                # Clear existing cache for create_or_load_uploaded_retriever
                create_or_load_uploaded_retriever.clear()
                # Pass the hashes to trigger rebuild logic inside the function
                st.session_state.current_retriever = create_or_load_uploaded_retriever(
                    [f.file_id for f in uploaded_files]
                    if hasattr(uploaded_files[0], "file_id")
                    else [f.name for f in uploaded_files]
                )
                if st.session_state.current_retriever:
                    st.session_state.app_graph = compile_graph(
                        st.session_state.current_retriever
                    )
                    st.session_state.uploaded_docs_processed = True
                else:
                    st.error("Failed to process uploaded files.")
                st.session_state.messages = (
                    []
                )  # Clear messages after processing new files
        elif st.session_state.get("uploaded_docs_processed"):
            # If files were processed in a previous run and mode is still upload, try to load
            # Pass an empty list or None to indicate no new files for processing, just loading
            st.session_state.current_retriever = create_or_load_uploaded_retriever(None)
            if st.session_state.current_retriever:
                st.session_state.app_graph = compile_graph(
                    st.session_state.current_retriever
                )
    else:  # Use existing persistent database (or rely on web search)
        st.session_state.current_retriever = get_persistent_chroma_retriever()
        st.session_state.app_graph = compile_graph(st.session_state.current_retriever)


# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "documents" in message and message["documents"]:
            with st.expander("Sources"):
                for doc in message["documents"]:
                    # Display source if available in metadata
                    source_info = doc.metadata.get("source", "N/A")
                    st.info(f"**Source:** {source_info}\n\n{doc.page_content}")


# Accept user input
if prompt := st.chat_input("Ask me a question!"):
    if not st.session_state.app_graph:
        st.warning(
            "Please select a knowledge base and ensure it's ready before asking questions."
        )
        st.stop()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        final_state = {}
        # Use st.status to show the thinking process
        with st.status("🧠 **Thinking...**", expanded=True) as status:
            inputs = {"question": prompt}
            try:
                # Stream the output from the graph
                for output in st.session_state.app_graph.stream(
                    inputs, {"recursion_limit": 10}
                ):
                    for key, value in output.items():
                        final_state.update(value)
                        status.write(f"**Node '{key}':** Executing...")
                        # st.write(value) # Uncomment to see full state at each step
            except Exception as e:
                st.error(f"An error occurred during graph execution: {e}")
                status.update(label="❌ Error", state="error", expanded=False)
                st.stop()

        # Display the final answer
        final_generation = final_state.get(
            "generation", "Sorry, I couldn't find an answer."
        )
        st.markdown(final_generation)

        # Display source documents in an expander
        final_documents = final_state.get("documents", [])
        if final_documents:
            with st.expander("Sources"):
                for doc in final_documents:
                    source_info = doc.metadata.get("source", "N/A")
                    st.info(f"**Source:** {source_info}\n\n{doc.page_content}")

        # Add assistant response to chat history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_generation,
                "documents": final_documents,
            }
        )
