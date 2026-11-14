from pathlib import Path
from langchain_core.documents import Document


def load_project_files(folder_path: str | Path) -> list[Document]:
    path = Path(folder_path)
    extensions = {".py", ".java", ".properties"}
    documents = []

    for file_path in path.rglob("*"):
        if file_path.suffix in extensions:
            text = file_path.read_text(encoding="utf-8")
            documents.append(
                Document(page_content=text, metadata={"source": str(file_path)})
            )

    return documents
