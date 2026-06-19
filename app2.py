import os
import json
import tempfile
import shutil
import fitz
from pathlib import Path
import streamlit as st
import chromadb
from dotenv import load_dotenv
from PIL import Image

from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import MarkdownNodeParser
from llama_parse import LlamaParse

from reranking import ReRanking
from metadata_ext import metadata_extraction
from image_link import build_image_index, attach_images_to_nodes, enrich_nodes_with_image_summaries
from image_metadata import extract_image_metadata
from chunk_merger import merge_small_chunks_with_overlap
from summarization import summarize_images_and_update_metadata

# Page configuration
st.set_page_config(
    page_title="Document Retrieval System",
    page_icon="🔍",
    layout="wide"
)

# Load environment variables
load_dotenv()

CHROMA_DIR = os.path.abspath("./chromadb")
COLLECTION_NAME = "Chunks_Final"
UPLOAD_DIR = os.path.abspath("./uploaded_pdfs")
UPLOAD_METADATA_DIR = os.path.abspath("./uploaded_metadata")
EXTRACTED_IMAGES_DIR = os.path.abspath("./extracted_images")

# Create upload directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UPLOAD_METADATA_DIR, exist_ok=True)
os.makedirs(EXTRACTED_IMAGES_DIR, exist_ok=True)

# Configuration
MIN_CHUNK_CHARS = 800
MAX_CHUNK_CHARS = 2000
OVERLAP_CHARS = 250

# Initialize the system (cached to avoid reloading)
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
        st.warning("Chroma collection is empty! Upload documents to get started.")
        return None, 0
    
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model
    )
    
    retriever = index.as_retriever(similarity_top_k=5)
    
    return retriever, collection.count()

def extract_images_from_pdf(pdf_path, output_folder=None):
    """Extract all images from PDF"""
    if output_folder is None:
        output_folder = EXTRACTED_IMAGES_DIR
    
    os.makedirs(output_folder, exist_ok=True)
    
    pdf_id = Path(pdf_path).stem.lower().replace(" ", "_")
    doc = fitz.open(pdf_path)
    image_info = []
    
    print(f"Scanning {len(doc)} pages for images...")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            image_path = f"{output_folder}/{pdf_id}_p{page_num+1}_img_{img_index+1}.png"
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            image_info.append({
                "path": image_path,
                "page": page_num + 1,
                "index": img_index + 1
            })
    
    doc.close()
    return image_info

def process_uploaded_pdf(uploaded_file, progress_bar, status_text):
    """Process a single uploaded PDF and add it to the vector store"""
    try:
        # Save uploaded file temporarily
        temp_pdf_path = Path(UPLOAD_DIR) / uploaded_file.name
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        status_text.text(f"📄 Parsing {uploaded_file.name}...")
        progress_bar.progress(20)
        
        # Check for API key
        llama_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not llama_key:
            st.error("LLAMA_CLOUD_API_KEY not found in environment")
            return False, 0
        
        # Parse PDF
        parser = LlamaParse(
            result_type="markdown",
            api_key=llama_key,
            verbose=False,
            extract_layout=True,
        )
        
        documents = SimpleDirectoryReader(
            input_files=[str(temp_pdf_path)],
            file_extractor={".pdf": parser}
        ).load_data()
        
        status_text.text("✂️ Creating chunks...")
        progress_bar.progress(40)
        
        # Create and merge chunks
        node_parser = MarkdownNodeParser()
        text_nodes = node_parser.get_nodes_from_documents(documents)
        
        text_nodes = merge_small_chunks_with_overlap(
            text_nodes,
            min_chars=MIN_CHUNK_CHARS,
            max_chars=MAX_CHUNK_CHARS,
            overlap_chars=OVERLAP_CHARS,
        )
        
        # Normalize page metadata
        for node in text_nodes:
            page_label = node.metadata.get("page_label")
            if page_label:
                try:
                    node.metadata["page"] = int(page_label)
                except ValueError:
                    pass
        
        status_text.text("🖼️ Processing images...")
        progress_bar.progress(60)
        
        # Extract images from PDF
        
        try:
            image_info = extract_images_from_pdf(temp_pdf_path)
            st.info(f"Extracted {len(image_info)} images from {uploaded_file.name}")
        except Exception as e:
            st.warning(f"Could not extract images: {str(e)}")
            image_info = []
        
        # Process image metadata
        metadata_file = Path(UPLOAD_METADATA_DIR) / f"{temp_pdf_path.stem}.json"
        try:
            metadata_extraction(temp_pdf_path, metadata_file)
            
            with open(metadata_file, "r", encoding="utf-8") as f:
                layout_json = json.load(f)
            
            image_metadata = extract_image_metadata(layout_json, temp_pdf_path)
            
            if image_metadata:
                image_index = build_image_index(image_metadata)
                attach_images_to_nodes(text_nodes, image_index)
                text_nodes = enrich_nodes_with_image_summaries(text_nodes)
        except Exception as e:
            st.warning(f"Could not process image metadata: {str(e)}")
        
        status_text.text("💾 Indexing to vector store...")
        progress_bar.progress(80)
        
        # Add to vector store
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index and add nodes
        VectorStoreIndex(
            nodes=text_nodes,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        return True, len(text_nodes)
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False, 0

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

# Header
st.title("🔍 Document Retrieval System")
st.markdown("Upload PDFs and retrieve relevant information using AI-powered search")

# Sidebar for document management
with st.sidebar:
    st.header("📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to add to the knowledge base"
    )
    
    if uploaded_files:
        if st.button("Process Uploaded PDFs", type="primary"):
            total_files = len(uploaded_files)
            total_chunks = 0
            successful = 0
            
            for idx, uploaded_file in enumerate(uploaded_files, 1):
                st.markdown(f"**Processing {idx}/{total_files}: {uploaded_file.name}**")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                success, num_chunks = process_uploaded_pdf(
                    uploaded_file, 
                    progress_bar, 
                    status_text
                )
                
                if success:
                    successful += 1
                    total_chunks += num_chunks
                    st.success(f"✅ {uploaded_file.name}: {num_chunks} chunks added")
                else:
                    st.error(f"❌ Failed to process {uploaded_file.name}")
                
                st.markdown("---")
            
            st.success(f"🎉 Processed {successful}/{total_files} files successfully! Added {total_chunks} total chunks.")
            
            # Clear cache to reload retriever
            st.cache_resource.clear()
            st.rerun()
    
    st.markdown("---")
    
    st.header("ℹ️ About")
    st.markdown("""
    This app retrieves relevant document chunks from your vector database.
    
    **Features:**
    - 📤 Upload and process PDFs
    - 🔍 Vector similarity search
    - 🎯 Reranking for improved relevance
    - 📄 Preview of source documents
    - 🖼️ Image display with summaries
    
    **How to use:**
    1. Upload PDFs using the sidebar
    2. Enter your query below
    3. Click Search to find relevant content
    4. View results in organized tabs
    """)
    
    st.header("⚙️ Current Settings")
    
    # Try to get collection info
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        doc_count = collection.count()
        st.metric("Documents in Collection", doc_count)
    except:
        st.metric("Documents in Collection", "0")
    
    st.markdown(f"""
    - **Collection:** {COLLECTION_NAME}
    - **Top-K Results:** 5
    - **Reranked Results:** 3
    - **Embedding:** all-MiniLM-L6-v2
    - **Chunk Size:** {MIN_CHUNK_CHARS}-{MAX_CHUNK_CHARS} chars
    - **Overlap:** {OVERLAP_CHARS} chars
    """)
    
    # Option to clear collection
    st.markdown("---")
    st.header("🗑️ Management")
    
    # Use session state to manage confirmation
    if 'confirm_delete' not in st.session_state:
        st.session_state.confirm_delete = False
    
    if st.button("Clear All Documents", type="secondary"):
        st.session_state.confirm_delete = True
    
    if st.session_state.confirm_delete:
        st.warning("⚠️ This will delete all documents permanently!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm", type="primary"):
                try:
                    client = chromadb.PersistentClient(path=CHROMA_DIR)
                    client.delete_collection(name=COLLECTION_NAME)
                    st.success("Collection cleared successfully!")
                    st.session_state.confirm_delete = False
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing collection: {str(e)}")
        with col2:
            if st.button("❌ Cancel"):
                st.session_state.confirm_delete = False
                st.rerun()

# Initialize retriever
retriever, doc_count = initialize_retriever()

if retriever is None:
    st.info("👈 Upload some PDFs using the sidebar to get started!")
    st.stop()
else:
    st.success(f"✅ System ready with {doc_count} documents in collection")

# Query input using form to ensure proper submission
with st.form(key="query_form"):
    query = st.text_input(
        "Enter your query:",
        placeholder="What would you like to know?",
        help="Enter a question or search query to find relevant document chunks",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.form_submit_button("🔍 Search", type="primary", use_container_width=True)

# Process search when form is submitted
if submit_button and query:
    with st.spinner("🔍 Retrieving and reranking documents..."):
        try:
            # Retrieve results
            results = retriever.retrieve(query)
            
            # Rerank results
            reranked_chunks = ReRanking(query, results, 3)
            
            # Display results in tabs
            tab1, tab2 = st.tabs(["🎯 Reranked Results", "📄 All Retrieved Chunks"])
            
            # Tab 1: Reranked Results
            with tab1:
                st.subheader("Top Reranked Results")
                st.markdown("*Documents reordered by relevance using reranking model*")
                st.markdown(f"**Query:** _{query}_")
                st.markdown("---")
                
                if reranked_chunks:
                    for i, (node, score) in enumerate(reranked_chunks, 1):
                        with st.expander(f"🏆 Result #{i} - Relevance Score: {score:.4f}", expanded=(i==1)):
                            st.markdown(f"**Reranking Score:** `{score:.4f}`")
                            
                            # Find original node
                            original_node = None
                            for res_node in results:
                                if res_node.get_content() == node[1]:
                                    original_node = res_node
                                    break
                            
                            if original_node and original_node.metadata.get("type") == "image":
                                # Display image
                                image_path = original_node.metadata.get("image_path")
                                if image_path:
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        st.markdown("**📸 Image:**")
                                        img = display_image_from_path(image_path)
                                        if img:
                                            st.image(img, use_container_width=True)
                                        
                                        # Display metadata
                                        st.markdown(f"**📄 Page:** {original_node.metadata.get('page', 'N/A')}")
                                        st.markdown(f"**📑 Source:** {original_node.metadata.get('source_pdf', 'N/A')}")
                                    
                                    with col2:
                                        st.markdown("**📝 Image Summary:**")
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
                                # Regular text content
                                if original_node:
                                    st.markdown(f"**📑 Source:** {original_node.metadata.get('file_name', 'Unknown')}")
                                    st.markdown(f"**📄 Page:** {original_node.metadata.get('page', 'N/A')}")
                                
                                st.markdown("**📝 Content:**")
                                st.text_area(
                                    "Content",
                                    value=node[1],
                                    height=200,
                                    key=f"reranked_{i}_{hash(query)}",
                                    label_visibility="collapsed"
                                )
                else:
                    st.info("No reranked results available")
            
            # Tab 2: Retrieved Chunks
            with tab2:
                st.subheader("All Retrieved Chunks")
                st.markdown("*Original retrieval results before reranking*")
                st.markdown(f"**Query:** _{query}_")
                st.markdown("---")
                
                for i, node in enumerate(results, 1):
                    with st.expander(f"Chunk {i} - Similarity: {node.score:.4f}", expanded=False):
                        # Check if this is an image node
                        if node.metadata.get("type") == "image":
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.metric("Similarity Score", f"{node.score:.4f}")
                                st.markdown(f"**Type:** 🖼️ Image")
                                st.markdown(f"**Page:** {node.metadata.get('page', 'N/A')}")
                                st.markdown(f"**Source PDF:** {node.metadata.get('source_pdf', 'N/A')}")
                                
                                # Display image
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
                            # Regular text content
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                st.metric("Similarity Score", f"{node.score:.4f}")
                                st.markdown(f"**Type:** 📄 Text")
                                st.markdown(f"**Source:** {node.metadata.get('file_name', 'Unknown')}")
                                st.markdown(f"**Page:** {node.metadata.get('page', 'N/A')}")
                            
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
            
            st.success(f"✅ Found {len(results)} results, showing top 3 reranked")
            
        except Exception as e:
            st.error(f"❌ Error during retrieval: {str(e)}")
            st.exception(e)
            
elif submit_button and not query:
    st.warning("⚠️ Please enter a query to search")