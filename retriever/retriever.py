"""
Система RAG (Retrieval-Augmented Generation) для Автоасессора.

Основные классы:
- EnhancedBM25: BM25 с нормализацией (RRF, перцентильная)
- QuestionAnswerRetriever: гибридный поиск (FAISS + BM25) для примеров
- DomainRetriever: поиск по доменным знаниям

Гибридный поиск объединяет плотный поиск (FAISS) и разреженный (BM25).
"""

import logging
from typing import Dict, List

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from scipy.stats import rankdata
from retriever.doc_loaders import load_documents_from_directory

# =====================================================
# CONSTANTS
# =====================================================

# BM25 parameters
BM25_K1: float = 1.5
BM25_B: float = 0.75

# RRF normalization parameter
RRF_K: int = 60

# Default hybrid search weight
DEFAULT_BETA: float = 0.5


# =====================================================
# ENHANCED BM25
# =====================================================


class EnhancedBM25(BM25Okapi):
    """Улучшенный BM25 с опциями нормализации: RRF и Перцентильная."""

    def __init__(self, corpus, k1: float = BM25_K1, b: float = BM25_B):
        super().__init__(corpus, k1=k1, b=b)

    def get_normalized_scores(self, query, method="rrf", k: int = RRF_K, percentile: int = 95):
        """
        Возвращает нормализованные скоры от 0 до 1.

        Аргументы:
            query: список токенов запроса.
            method: 'percentile' или 'rrf'.
            k: параметр сглаживания для RRF (обычно 60).
            percentile: перцентиль для отсечения выбросов (от 0 до 100).
        """
        raw_scores = self.get_scores(query)

        if len(raw_scores) == 0:
            return np.array([])

        if method == "rrf":
            return self._rrf_normalization(raw_scores, k=k)
        elif method == "percentile":
            return self._percentile_normalization(raw_scores, percentile=percentile)
        else:
            raise ValueError(f"Неизвестный метод нормализации: {method}")

    def _rrf_normalization(self, scores, k=60):
        """
        Reciprocal Rank Fusion нормализация.
        Преобразует скоры в ранги и применяет формулу RRF.
        Устойчива к выбросам, так как зависит только от порядка (ранга), а не величины скора.
        """
        # Одинаковая релевантность не должна зависеть от позиции документа в корзине.
        ranks = rankdata(-np.asarray(scores), method="average")

        # Формула RRF: 1 / (k + rank)
        rrf_scores = 1.0 / (k + ranks)

        # Нормализуем к диапазону [0, 1] для удобства
        # Максимальное значение RRF будет у первого ранга: 1 / (k + 1)
        max_rrf = 1.0 / (k + 1)
        min_rrf = 1.0 / (k + len(scores))

        if max_rrf - min_rrf > 0:
            return (rrf_scores - min_rrf) / (max_rrf - min_rrf)
        else:
            return np.ones_like(scores) * 0.5

    def _percentile_normalization(self, scores, percentile=95):
        """
        Перцентильная нормализация.
        Считает нормализацию не от Min/Max, а от перцентилей (например, 5-го и 95-го).
        Это позволяет "игнорировать" экстремальные выбросы.
        """
        # Вычисляем перцентили
        p_low = np.percentile(scores, 100 - percentile)
        p_high = np.percentile(scores, percentile)

        # Если данные слишком однородны (p_high == p_low), возвращаем нейтральное значение
        if p_high - p_low < 1e-9:
            # Если все скоры одинаковы
            if p_high == np.max(scores):
                return np.ones_like(scores)
            return np.ones_like(scores) * 0.5

        # Нормализуем с обрезкой (clipping)
        # Значения ниже p_low становятся 0, выше p_high - 1
        normalized = (scores - p_low) / (p_high - p_low)

        # Обрезаем значения, вышедшие за пределы [0, 1] из-за выбросов
        return np.clip(normalized, 0, 1)


# =====================================================
# BASE RETRIEVER
# =====================================================


class BaseRetriever:
    """Базовый класс ретривера с общей логикой"""

    def __init__(self, embedding_model, max_length: int = 512):
        self._embedding_model = embedding_model
        self._max_length = max_length
        self.logger = logging.getLogger(self.__class__.__name__)

    def _merge_results(
        self, vector_results: List[Dict], bm25_results: List[Dict], beta: float = DEFAULT_BETA
    ) -> List[Dict]:
        """Объединяет результаты векторного и BM25 поиска с учетом весов (beta)"""
        all_results = {}
        alpha = 1.0 - beta  # Вес для embedding scores

        # Добавляем векторные результаты
        for result in vector_results:
            key = f"{result.get('source', '')}_{result.get('chunk_id', 0)}"
            score = result.get("similarity_score", 0)
            all_results[key] = {
                "content": result["content"],
                "source": result.get("source", ""),
                "metadata": result.get("metadata", {}),
                "similarity_score": score,
                "bm25_score": 0,
                "total_score": alpha * score,
            }

        # Добавляем BM25 результаты
        for result in bm25_results:
            key = f"{result.get('source', '')}_{result.get('chunk_id', 0)}"
            score = result.get("bm25_score", 0)
            if key in all_results:
                all_results[key]["bm25_score"] = score
                all_results[key]["total_score"] += beta * score
            else:
                all_results[key] = {
                    "content": result["content"],
                    "source": result.get("source", ""),
                    "metadata": result.get("metadata", {}),
                    "similarity_score": 0,
                    "bm25_score": score,
                    "total_score": beta * score,
                }

        # Сортировка по общему скору
        return sorted(
            all_results.values(), key=lambda x: x["total_score"], reverse=True
        )


# =====================================================
# QUESTION-ANSWER RETRIEVER
# =====================================================


class QuestionAnswerRetriever(BaseRetriever):
    """
    Ретривер для поиска по примерам вопросов и ответов

    На вход подаётся list[dict] с примерами:
    examples = [
        {"question": "Какие условия по кредиту?", "answer": "Процентная ставка х% на год..."},
        ...
    ]
    """

    def __init__(
        self, embedding_model, examples: List[Dict[str, str]], max_length: int = 512
    ):
        super().__init__(embedding_model, max_length)

        self._examples = examples
        self._documents = []
        self._questions_for_bm25 = []

        self.logger.info("Инициализация QuestionAnswerRetriever")
        self.logger.info(f"Количество примеров: {len(examples)}")

        # Оценка относится ко всему примеру, включая поздние реплики диалога.
        for item in examples:
            question = item["question"]
            answer = item["answer"]
            doc = Document(
                page_content=question,
                metadata={
                    "answer": answer,
                    "question": question,
                },
            )
            self._documents.append(doc)
            self._questions_for_bm25.append(question)

        # Создаем векторное хранилище с FAISS (без HNSW)
        self._vectorstore = FAISS.from_documents(
            documents=self._documents,
            embedding=self._embedding_model,
            normalize_L2=True,
        )

        # Создаем BM25 индекс
        tokenized_corpus = [q.lower().split() for q in self._questions_for_bm25]
        self._bm25 = EnhancedBM25(tokenized_corpus)

        self.logger.info("QuestionAnswerRetriever инициализирован с FAISS")

    def hybrid_search(self, query: str, k: int = 10, beta: float = 0.8) -> List[Dict]:
        """
        Гибридный поиск по примерам вопросов и ответов
        beta: вес для BM25 (от 0 до 1). Вес для embeddings = 1 - beta.
        k=10: при несбалансированных метках больше примеров устойчивее передают
        политику разметки судье (замер на CI09877398).
        """
        alpha = 1.0 - beta

        # 1. Векторный поиск
        results = self._vectorstore.similarity_search_with_relevance_scores(
            query, k=k
        )

        vector_results = [
            {
                "question": doc.metadata["question"],
                "answer": doc.metadata["answer"],
                "similarity_score": float(score),
                "source": "vector_search",
            }
            for doc, score in results
        ]

        # 2. BM25 поиск
        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_normalized_scores(tokenized_query)

        bm25_results = []
        if len(bm25_scores) > 0:
            top_indices = np.argsort(bm25_scores)[::-1][:k]

            for idx in top_indices:
                bm25_results.append(
                    {
                        "question": self._questions_for_bm25[idx],
                        "answer": self._examples[idx]["answer"],
                        "bm25_score": float(bm25_scores[idx]),
                        "source": "bm25_search",
                    }
                )

        # 3. Объединение результатов с взвешиванием
        all_results = {}

        # Добавляем векторные результаты
        for result in vector_results:
            key = result["question"]
            score = result["similarity_score"]
            all_results[key] = {
                "question": result["question"],
                "answer": result["answer"],
                "similarity_score": score,
                "bm25_score": 0,
                "total_score": alpha * score,
            }

        # Добавляем BM25 результаты
        for result in bm25_results:
            key = result["question"]
            score = result["bm25_score"]
            if key in all_results:
                all_results[key]["bm25_score"] = score
                all_results[key]["total_score"] += beta * score
            else:
                all_results[key] = {
                    "question": result["question"],
                    "answer": result["answer"],
                    "similarity_score": 0,
                    "bm25_score": score,
                    "total_score": beta * score,
                }

        # 4. Сортировка
        sorted_results = sorted(
            all_results.values(), key=lambda x: x["total_score"], reverse=True
        )[:k]
        logging.debug(
            f"Полученные из retriever чанки: {[(val['similarity_score'], val['bm25_score']) for val in sorted_results]}"
        )
        return sorted_results


# =====================================================
# DOMAIN RETRIEVER
# =====================================================


class DomainRetriever(BaseRetriever):
    """
    Ретривер для поиска по содержимому документов со знаниями из доменной области

    На вход подаётся list[Document] с документами:
    documents = [
        Document(page_content="Кредиты выдаются под х% годовых для...", metadata={"source": "file1.pdf"}),
        ...
    ]
    """

    def __init__(
        self, embedding_model, documents: List[Document], max_length: int = 512
    ):
        super().__init__(embedding_model, max_length)

        self._documents = documents
        self.logger.info(
            f"Инициализация ContentRetriever с {len(documents)} документами"
        )

        # Создаем векторное хранилище с FAISS (без HNSW)
        self._vectorstore = FAISS.from_documents(
            documents=self._documents,
            embedding=self._embedding_model,
            normalize_L2=True,
        )

        # Подготавливаем корпус для BM25
        self._bm25_corpus = [doc.page_content for doc in self._documents]
        tokenized_corpus = [text.lower().split() for text in self._bm25_corpus]
        self._bm25 = EnhancedBM25(tokenized_corpus)

        self.logger.info("ContentRetriever инициализирован с FAISS")

    def hybrid_search(self, query: str, k: int = 5, beta: float = 0.5) -> List[Dict]:
        """
        Гибридный поиск по содержимому документов
        beta: вес для BM25 (от 0 до 1).
        """
        # 1. Векторный поиск
        results = self._vectorstore.similarity_search_with_relevance_scores(
            query[: self._max_length], k=k
        )
        vector_results = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", 0),
                "metadata": doc.metadata,
                "similarity_score": float(score),
            }
            for doc, score in results
        ]

        # 2. BM25 поиск
        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_normalized_scores(tokenized_query)

        bm25_results = []
        if len(bm25_scores) > 0:
            top_indices = np.argsort(bm25_scores)[::-1][:k]

            for idx in top_indices:
                doc = self._documents[idx]
                bm25_results.append(
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "chunk_id": doc.metadata.get("chunk_id", 0),
                        "metadata": doc.metadata,
                        "bm25_score": float(bm25_scores[idx]),
                    }
                )

        # 3. Объединение результатов с использованием базового метода
        return self._merge_results(vector_results, bm25_results, beta)[:k]


# Фабричная функция для удобного создания ретриверов
def create_retriever(
    retriever_type: str, embedding_model, data, max_length: int = 512, **kwargs
):
    if retriever_type == "question_answer":
        if not isinstance(data, list) or not all(
            isinstance(item, dict) for item in data
        ):
            raise ValueError(
                "Для question_answer retriever нужен list[dict] с вопросами и ответами"
            )

        return QuestionAnswerRetriever(
            embedding_model=embedding_model, examples=data, max_length=max_length
        )

    elif retriever_type == "content":
        # Если передан путь к директории
        if isinstance(data, str):
            chunk_size = kwargs.get("chunk_size", 1000)
            chunk_overlap = kwargs.get("chunk_overlap", 200)

            documents = load_documents_from_directory(
                directory_path=data, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        # Если передан список документов
        elif isinstance(data, list) and all(
            isinstance(item, Document) for item in data
        ):
            documents = data
        else:
            raise ValueError(
                "Для content retriever нужен list[Document] или путь к директории"
            )

        return DomainRetriever(
            embedding_model=embedding_model, documents=documents, max_length=max_length
        )

    else:
        raise ValueError(f"Неизвестный тип ретривера: {retriever_type}")
