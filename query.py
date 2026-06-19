import os
import chromadb
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from reranking import ReRanking

load_dotenv()

CHROMA_DIR = os.path.abspath("./chromadb")
COLLECTION_NAME = "Chunks_3"

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)

print("COLLECTION COUNT:", collection.count())
assert collection.count() > 0, "Chroma collection is empty!"

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=Settings.embed_model
)

retriever = index.as_retriever(
    similarity_top_k=5
)

while True: 
    query = input("Enter your query: ") 

    if query == "E" or query == "e" :
        break

    results = retriever.retrieve(query)

    reranked_chunks = ReRanking(query, results, 3)

    print("\RETRIEVED CHUNKS:\n")

    for i, node in enumerate(results, 1):
        print(f"\n--- RESULT {i} ---")
        print("Score:", node.score)
        print("Source:", node.metadata.get("file_name"))
        print(node.get_content()[:800])

    for i, (node, score) in enumerate(reranked_chunks, 1):
        print(f"\n--- RESULT {i} ---")
        print("Rerank Score:", score)
        print("Vector Score:", node.score)
        print("Content:\n", node.get_content())


