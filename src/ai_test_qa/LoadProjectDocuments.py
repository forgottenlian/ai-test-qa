from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


def load_project_files(folder_path: str | Path) -> list[Document]:
    path = Path(folder_path)
    extensions = {
        ".py",
        ".java",
        ".properties",
        ".xhtml",
        ".html",
        ".xml",
        ".json",
        ".yaml",
        ".yml",
        ".txt",
    }
    documents = []

    for file_path in path.rglob("*"):
        if file_path.suffix in extensions:
            print(f"Loading file: {file_path}")
            text = file_path.read_text(encoding="utf-8")
            documents.append(
                Document(page_content=text, metadata={"source": str(file_path)})
            )

    return documents


def save_documents_to_vectorstore(
    documents: list[Document], vectorstore_path: str | Path
) -> None:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=10,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name=str(Path.home() / "AI-models" / "all-MiniLM-L6-v2")
    )
    print(f"Created embeddings using model: {embeddings.model_name}")

    vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
    vectorstore.dump(vectorstore_path)
    print(f"Saved vector store to '{vectorstore_path}'")
