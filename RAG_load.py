from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# 1. Load the exact same embedding model used when saving
embeddings = HuggingFaceEmbeddings(
    model_name=str(Path.home() / "AI-models" / "all-MiniLM-L6-v2")
)

# 2. Load the saved vector store from file
vectorstore = InMemoryVectorStore.load("vectorstore.json", embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 3. Query immediately without re-indexing files
results = retriever.invoke("How do I run this program?")
for doc in results:
    print(doc.page_content)
