from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


def load_text_file(file_path: str | Path) -> Document:
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    return Document(page_content=text, metadata={"source": str(path)})


docs = []
docs.append(load_text_file(Path("src") / "ai_test_qa" / "__init__.py"))
docs.append(load_text_file(Path("README.md")))
print(f"Loaded {len(docs)} documents")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=10,
)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

embeddings = HuggingFaceEmbeddings(
    model_name=str(Path.home() / "AI-models" / "all-MiniLM-L6-v2")
)
print(f"Created embeddings using model: {embeddings.model_name}")

vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
vectorstore.dump("vectorstore.json")
print("Saved vector store to 'vectorstore.json'")
