from pathlib import Path
from typing import Any, List, Union

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def load_all_documents(data_dir: Union[Path, str, None] = None) -> List[Any]:
    """
    Load all supported documents from the data directory and convert to LangChain Document objects.
    Supported document types: PDF, CSV, TXT, and XLSX.
    """
    if data_dir is None:
        data_path = DEFAULT_DATA_DIR
    else:
        data_path = Path(data_dir)
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path

    data_path = data_path.resolve()
    print(f"Loading documents from: {data_path}")

    if not data_path.exists():
        print(f"Data directory not found: {data_path}")
        return []

    documents: List[Any] = []

    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files: {[str(file) for file in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"Loading PDF: {pdf_file}")
        try:
            loaded = PyPDFLoader(str(pdf_file)).load()
            documents.extend(loaded)
            print(f"Loaded {len(loaded)} pages from {pdf_file.name}")
        except Exception as e:
            print(f"Error loading PDF {pdf_file.name}: {e}")

    text_files = list(data_path.glob("**/*.txt"))
    print(f"Found {len(text_files)} text files: {[str(file) for file in text_files]}")
    for text_file in text_files:
        print(f"Loading text: {text_file}")
        try:
            loaded = TextLoader(str(text_file), encoding="utf-8").load()
            documents.extend(loaded)
            print(f"Loaded {len(loaded)} documents from {text_file.name}")
        except Exception as e:
            print(f"Error loading text {text_file.name}: {e}")

    csv_files = list(data_path.glob("**/*.csv"))
    print(f"Found {len(csv_files)} CSV files: {[str(file) for file in csv_files]}")
    for csv_file in csv_files:
        print(f"Loading CSV: {csv_file}")
        try:
            loaded = CSVLoader(str(csv_file)).load()
            documents.extend(loaded)
            print(f"Loaded {len(loaded)} rows from {csv_file.name}")
        except Exception as e:
            print(f"Error loading CSV {csv_file.name}: {e}")

    excel_files = list(data_path.glob("**/*.xlsx"))
    print(f"Found {len(excel_files)} Excel files: {[str(file) for file in excel_files]}")
    for excel_file in excel_files:
        print(f"Loading Excel: {excel_file}")
        try:
            loaded = UnstructuredExcelLoader(str(excel_file)).load()
            documents.extend(loaded)
            print(f"Loaded {len(loaded)} sheets from {excel_file.name}")
        except Exception as e:
            print(f"Error loading Excel {excel_file.name}: {e}")

    return documents
