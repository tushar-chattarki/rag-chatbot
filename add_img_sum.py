import json
import chromadb
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

CHROMA_DIR = "./chromadb"
COLLECTION_NAME = "Chunks_Final"
IMAGE_METADATA_FILE = "Image_Metadata/image_metadata_final.json"

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_collection(
    name=COLLECTION_NAME
)

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=Settings.embed_model
)

print(f"Existing vectors in collection: {collection.count()}")


with open(IMAGE_METADATA_FILE, "r", encoding="utf-8") as f:
    image_metadata = json.load(f)


image_nodes = []

for img in image_metadata:
    summary = img.get("summary")
    if not summary:
        continue

    image_nodes.append(
        Document(
            text=summary,
            metadata={
                "type": "image",
                "image_id": img["image_id"],
                "image_path": img["image_path"],
                "page": img["page"],
                "source_pdf": img["source_pdf"],
            }
        )
    )

print(f"Image summaries to insert: {len(image_nodes)}")

if image_nodes:
    index.insert_nodes(image_nodes)
    print("Image summaries inserted successfully")
else:
    print("No image summaries found to insert")

print(f"TOTAL vectors after insert: {collection.count()}")
