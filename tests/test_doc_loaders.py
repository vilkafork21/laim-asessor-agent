"""Загрузчик доменного RAG: документы находятся и во вложенных каталогах."""
from __future__ import annotations

from retriever.doc_loaders import load_documents_from_directory


def test_nested_directories_are_loaded(tmp_path):
    # Архив с папками (типичный экспорт) проходил проверку по rglob, а
    # загрузчик смотрел только верхний уровень — RAG молча отключался.
    (tmp_path / "top.txt").write_text("Верхний документ", encoding="utf-8")
    nested = tmp_path / "раздел" / "глубже"
    nested.mkdir(parents=True)
    (nested / "inner.txt").write_text("Вложенный документ", encoding="utf-8")

    documents = load_documents_from_directory(str(tmp_path), chunk_size=0)

    assert sorted(doc.metadata["source"] for doc in documents) == ["inner.txt", "top.txt"]
