"""
Загрузчики документов для RAG системы.

Основные функции:
- load_instruction: чтение инструкций (PDF, DOCX, TXT)
- load_documents_from_directory: загрузка и разбиение документов на чанки

Поддерживаемые форматы: PDF, DOCX, DOC, TXT, MD, CSV
"""

import logging
from pathlib import Path
from typing import Dict, List, Type

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Supported file extensions and their loaders
SUPPORTED_EXTENSIONS: Dict[str, Type] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".csv": TextLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
}


# =====================================================
# INSTRUCTION LOADER
# =====================================================


def load_instruction(instruction_path: str) -> str:
    """Читает инструкцию из файла (PDF или DOCX)."""
    if instruction_path.endswith(".pdf"):
        import fitz  # pymupdf

        doc = fitz.open(instruction_path)
        instruction = "".join(page.get_text() for page in doc)
        logger.debug("Считана PDF инструкция")

    elif instruction_path.endswith(".docx"):
        from docx import Document

        doc = Document(instruction_path)
        instruction = "\n".join(para.text for para in doc.paragraphs)
        logger.debug("Считана DOCX инструкция")

    else:
        instruction = ""
        logger.error(f"Неподдерживаемый формат инструкции: {instruction_path}")

    return instruction


# =====================================================
# DOCUMENT DIRECTORY LOADER
# =====================================================


def load_documents_from_directory(
    directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """Загружает все документы из директории и разбивает на чанки."""
    try:  # langchain>=1.0 вынес сплиттеры в отдельный пакет
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        pass

    documents: List[Document] = []
    directory = Path(directory_path)

    if not directory.exists():
        raise ValueError(f"Директория {directory_path} не существует")

    # Collect all files
    all_files = []
    for ext in SUPPORTED_EXTENSIONS.keys():
        all_files.extend(directory.glob(f"*{ext}"))
        all_files.extend(directory.glob(f"*{ext.upper()}"))

    logger.info(f"Найдено {len(all_files)} файлов в директории {directory_path}")

    # Load each file
    for file_path in all_files:
        file_docs = _load_single_file(file_path)
        documents.extend(file_docs)

    # Split into chunks if needed
    if chunk_size > 0 and documents:
        documents = _split_into_chunks(documents, chunk_size, chunk_overlap)

    return documents


def _load_single_file(file_path: Path) -> List[Document]:
    """Загружает один файл и возвращает список документов."""
    file_ext = file_path.suffix.lower()

    if file_ext not in SUPPORTED_EXTENSIONS:
        logger.warning(f"Неподдерживаемый формат: {file_path.name}")
        return []

    try:
        loader_class = SUPPORTED_EXTENSIONS[file_ext]
        loader = loader_class(str(file_path))
        file_docs = loader.load()

        # Add source metadata
        for doc in file_docs:
            doc.metadata["source"] = file_path.name

        logger.info(f"Загружен файл: {file_path.name} ({len(file_docs)} документов)")
        return file_docs

    except Exception as e:
        logger.error(f"Ошибка загрузки файла {file_path.name}: {e}")
        return []


def _split_into_chunks(
    documents: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """Разбивает документы на чанки."""
    try:  # langchain>=1.0 вынес сплиттеры в отдельный пакет
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunked_docs = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        chunked_docs.extend(chunks)

    logger.info(f"Документы разбиты на {len(chunked_docs)} чанков")
    return chunked_docs