import argparse
from .LoadProjectDocuments import load_project_files


def main(argv: list[str] | None = None) -> None:
    print("Hello from ai-test-qa!")

    parser = argparse.ArgumentParser(description="AI Test QA")
    parser.add_argument("--folder", help="Path to the project folder")
    args = parser.parse_args(argv)

    if args.folder:
        load_project(args.folder)
    else:
        parser.print_help()


def load_project(folder_path: str) -> None:
    print(f"Loading project documents from '{folder_path}'...")
