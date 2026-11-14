from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


# -------------------------------------------------
# 1. Load your files (no TextLoader needed)
# -------------------------------------------------
def load_text_file(file_path: str | Path) -> Document:
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    return Document(page_content=text, metadata={"source": str(path)})


docs = []
docs.append(load_text_file(Path("src") / "ai_test_qa" / "__init__.py"))
docs.append(load_text_file(Path("README.md")))
print(f"Loaded {len(docs)} documents")

# -------------------------------------------------
# 2. Split into chunks
# -------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=10,
)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# -------------------------------------------------
# 3. Create embeddings (runs fully local, free)
# -------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name=str(Path.home() / "AI-models" / "all-MiniLM-L6-v2")
)

# -------------------------------------------------
# 4. Store in an in-memory vector store
# -------------------------------------------------
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})


# -------------------------------------------------
# 5. Ask questions (for now just show retrieved pieces)
# -------------------------------------------------
def ask(question: str):
    print(f"\nQuestion: {question}")
    print("=== Retrieved pieces ===")
    results = retriever.invoke(question)
    for doc in results:
        print(doc.page_content)
        print("---")


# Test it
ask("What does the main function do?")
ask("How do I run this program?")
ask("What is the purpose of this application?")
ask("What is the name of this application?")
