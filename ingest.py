import os
import json
import chromadb
from dotenv import load_dotenv
from pathlib import Path

from metadata_ext import metadata_extraction
from image_link import (
    build_image_index,
    attach_images_to_nodes,
    enrich_nodes_with_image_summaries
)
from image_metadata import extract_image_metadata
from chunk_merger import merge_small_chunks_with_overlap
from llama_parse import LlamaParse
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

PDF_DIR = "/Users/tusharc/Code/Python/RAG/RAG-Model/pdfs"
MET_DIR = "/Users/tusharc/Code/Python/RAG/RAG-Model/Metadata"
CHROMA_DIR = os.path.abspath("./chromadb")
COLLECTION_NAME = "Chunks_Final"
CHUNKS_DEBUG_FILE = "data/chunks.txt"

LLAMA_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(MET_DIR, exist_ok=True)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

vector_store = ChromaVectorStore(chroma_collection=collection)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

parser = LlamaParse(
    result_type="markdown",
    api_key=LLAMA_KEY,
    verbose=True,
    extract_layout=True,
)

documents = SimpleDirectoryReader(
    input_dir=PDF_DIR,
    file_extractor={".pdf": parser}
).load_data()

node_parser = MarkdownNodeParser()
text_nodes = node_parser.get_nodes_from_documents(documents)

print(f"Before merge: {len(text_nodes)} chunks")

text_nodes = merge_small_chunks_with_overlap(
    text_nodes,
    min_chars=800,
    max_chars=2000,
    overlap_chars=250,
)

print(f"After merge + overlap: {len(text_nodes)} chunks")

print(f"CHUNKS CREATED: {len(text_nodes)}")

for node in text_nodes:
    page_label = node.metadata.get("page_label")
    if page_label:
        try:
            node.metadata["page"] = int(page_label)
        except ValueError:
            pass

with open(CHUNKS_DEBUG_FILE, "w", encoding="utf-8") as f:
    for node in text_nodes:
        f.write(node.get_content())
        f.write("\n\n")

print(f"Debug chunks saved to {CHUNKS_DEBUG_FILE}")

image_metadata_all = []

for pdf_path in Path(PDF_DIR).glob("*.pdf"):
    metadata_file = f"{MET_DIR}/{pdf_path.stem}.json"

    metadata_extraction(pdf_path, metadata_file)

    with open(metadata_file, "r", encoding="utf-8") as f:
        layout_json = json.load(f)

    image_metadata = extract_image_metadata(layout_json, pdf_path)

    image_metadata_all.extend(image_metadata)


print(f"Loaded metadata for {len(image_metadata_all)} images")

image_index = build_image_index(image_metadata_all)

attached = attach_images_to_nodes(text_nodes, image_index)
print(f"Attached {attached} images to text nodes")

text_nodes = enrich_nodes_with_image_summaries(text_nodes)

if collection.count() == 0:
    VectorStoreIndex(
        nodes=text_nodes,
        storage_context=storage_context,
        embed_model=Settings.embed_model
    )
    print("Vector store created and persisted")
else:
    print("Collection already contains data — skipping ingestion")

print(f"TOTAL VECTORS IN COLLECTION: {collection.count()}")

assert collection.count() > 0, "Nothing was written to Chroma!"
