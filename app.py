import os
import streamlit as st
import chromadb
from dotenv import load_dotenv
from PIL import Image

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from reranking import ReRanking

st.set_page_config(
    page_title="Document Retrieval System",
    page_icon="🔍",
    layout="wide"
)

load_dotenv()

CHROMA_DIR = os.path.abspath("./chromadb")
COLLECTION_NAME = "Chunks_Final"

@st.cache_resource
def initialize_retriever():
    """Initialize and cache the retriever to avoid reloading on every interaction"""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    
    if collection.count() == 0:
        st.error("Chroma collection is empty! Please add documents first.")
        st.stop()
    
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model
    )
    
    retriever = index.as_retriever(similarity_top_k=5)
    
    return retriever, collection.count()

def display_image_from_path(image_path):
    """Load and display an image from the given path"""
    try:
        if os.path.exists(image_path):
            img = Image.open(image_path)
            return img
        else:
            st.warning(f"Image not found at: {image_path}")
            return None
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

st.title("Document Retrieval System")
st.markdown("Enter your query to retrieve and rerank relevant document chunks")

try:
    retriever, doc_count = initialize_retriever()
    st.success(f"✅ System initialized with {doc_count} documents in collection")
except Exception as e:
    st.error(f"Error initializing system: {str(e)}")
    st.stop()

with st.form(key="query_form"):
    query = st.text_input(
        "Enter your query:",
        placeholder="Type your question here...",
        help="Enter a question or search query to find relevant document chunks",
        key="query_input"
    )
    
    submit_button = st.form_submit_button("Search", type="primary")

if submit_button and query:
    with st.spinner("Retrieving and reranking documents..."):
        try:
            results = retriever.retrieve(query)

            reranked_chunks = ReRanking(query, results, 3)
            
            tab1, tab2 = st.tabs(["🎯 Reranked Results", "📄 Retrieved Chunks"])

            with tab1:
                st.subheader("Reranked Results")
                st.markdown("*Documents reordered by relevance using reranking model*")
                st.markdown(f"**Query:** _{query}_")
                
                if reranked_chunks:
                    for i, (node, score) in enumerate(reranked_chunks, 1):
                        with st.expander(f"Result {i} - Score: {score:.4f}", expanded=(i==1)):
                            st.markdown(f"**Reranking Score:** `{score:.4f}`")
                            
                            # Check if this is an image node
                            # node[0] contains the original node object
                            original_node = results[0] if i <= len(results) else None
                            for res_node in results:
                                if res_node.get_content() == node[1]:
                                    original_node = res_node
                                    break
                            
                            if original_node and original_node.metadata.get("type") == "image":
                                image_path = original_node.metadata.get("image_path")
                                if image_path:
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        st.markdown("**Image:**")
                                        img = display_image_from_path(image_path)
                                        if img:
                                            st.image(img, use_container_width=True)

                                        st.markdown(f"**Page:** {original_node.metadata.get('page', 'N/A')}")
                                        st.markdown(f"**Source PDF:** {original_node.metadata.get('source_pdf', 'N/A')}")
                                        st.markdown(f"**Image ID:** {original_node.metadata.get('image_id', 'N/A')}")
                                    
                                    with col2:
                                        st.markdown("**Image Summary:**")
                                        st.text_area(
                                            "Summary",
                                            value=node[1],
                                            height=300,
                                            key=f"reranked_img_{i}_{hash(query)}",
                                            label_visibility="collapsed"
                                        )
                                else:
                                    st.markdown("**Content:**")
                                    st.text_area(
                                        "Content",
                                        value=node[1],
                                        height=200,
                                        key=f"reranked_{i}_{hash(query)}",
                                        label_visibility="collapsed"
                                    )
                            else:
                                st.markdown("**Content:**")
                                st.text_area(
                                    "Content",
                                    value=node[1],
                                    height=200,
                                    key=f"reranked_{i}_{hash(query)}",
                                    label_visibility="collapsed"
                                )
                else:
                    st.info("No reranked results available")

            with tab2:
                st.subheader("Initial Retrieved Chunks")
                st.markdown("*Original retrieval results before reranking*")
                st.markdown(f"**Query:** _{query}_")
                
                for i, node in enumerate(results, 1):
                    with st.expander(f"Chunk {i} - Score: {node.score:.4f}", expanded=False):
                        if node.metadata.get("type") == "image":
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.metric("Similarity Score", f"{node.score:.4f}")
                                st.markdown(f"**Type:** 🖼️ Image")
                                st.markdown(f"**Page:** {node.metadata.get('page', 'N/A')}")
                                st.markdown(f"**Source PDF:** {node.metadata.get('source_pdf', 'N/A')}")
                                st.markdown(f"**Image ID:** {node.metadata.get('image_id', 'N/A')}")

                                st.markdown("**Image:**")
                                image_path = node.metadata.get("image_path")
                                if image_path:
                                    img = display_image_from_path(image_path)
                                    if img:
                                        st.image(img, use_container_width=True)
                            
                            with col2:
                                st.markdown("**Image Summary:**")
                                content = node.get_content()
                                st.text_area(
                                    "Summary",
                                    value=content,
                                    height=300,
                                    key=f"retrieved_img_{i}_{hash(query)}",
                                    label_visibility="collapsed"
                                )
                        else:
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                st.metric("Similarity Score", f"{node.score:.4f}")
                                st.markdown(f"**Type:** 📄 Text")
                                st.markdown(f"**Source:** {node.metadata.get('file_name', 'Unknown')}")
                            
                            with col2:
                                st.markdown("**Content Preview:**")
                                content = node.get_content()
                                st.text_area(
                                    "Content",
                                    value=content[:800] + ("..." if len(content) > 800 else ""),
                                    height=200,
                                    key=f"retrieved_{i}_{hash(query)}",
                                    label_visibility="collapsed"
                                )
            
            st.success(f"Found {len(results)} results for query: '{query}'")
            
        except Exception as e:
            st.error(f"Error during retrieval: {str(e)}")
            st.exception(e)
            
elif submit_button and not query:
    st.warning("Please enter a query to search")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app retrieves relevant document chunks from your vector database.
    
    **Features:**
    - Vector similarity search
    - Reranking for improved relevance
    - Preview of source documents
    - Image display with summaries
    
    **How to use:**
    1. Enter your query in the text box
    2. Click Search button
    3. View results in the tabs
    4. Images will be displayed alongside their summaries
    """)
    
    st.header("⚙️ Settings")
    st.markdown(f"""
    - **Collection:** {COLLECTION_NAME}
    - **Top-K Results:** 3
    - **Embedding Model:** all-MiniLM-L6-v2
    """)