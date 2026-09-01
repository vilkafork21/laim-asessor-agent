"""Длинный диалог целиком участвует в retrieval-эмбеддинге."""

from langchain_core.embeddings import Embeddings

from main import BoundedGigaChatEmbeddings
from retriever.retriever import QuestionAnswerRetriever


class RecordingEmbeddings:
    def __init__(self):
        self.document_chunks = []
        self.query_chunks = []

    def embed_documents(self, texts):
        self.document_chunks.extend(texts)
        return [[float(len(text)), float(text == "TAIL")] for text in texts]

    def embed_query(self, text):
        self.query_chunks.append(text)
        return [float(len(text)), float(text == "TAIL")]


def test_long_text_is_chunked_and_mean_pooled_to_one_vector():
    underlying = RecordingEmbeddings()
    embeddings = BoundedGigaChatEmbeddings(
        underlying,
        max_chars=4,
        batch_size=1,
    )

    document_vectors = embeddings.embed_documents(["abcdTAIL"])
    query_vector = embeddings.embed_query("abcdTAIL")

    assert underlying.document_chunks == ["abcd", "TAIL"]
    assert underlying.query_chunks == ["abcd", "TAIL"]
    assert document_vectors == [[4.0, 0.5]]
    assert query_vector == [4.0, 0.5]
    assert len(document_vectors) == 1


class RecordingVectorEmbeddings(Embeddings):
    def __init__(self):
        self.documents = []
        self.queries = []

    def embed_documents(self, texts):
        self.documents.extend(texts)
        return [[1.0, 0.0] for _text in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return [1.0, 0.0]


def test_retriever_does_not_cut_late_turn_before_bounded_embeddings():
    long_dialogue = "head" + "x" * 600 + "LATE_TURN"
    embeddings = RecordingVectorEmbeddings()
    retriever = QuestionAnswerRetriever(
        embedding_model=embeddings,
        examples=[{"question": long_dialogue, "answer": '{"score":1}'}],
    )

    retriever.hybrid_search(long_dialogue, k=1)

    assert embeddings.documents == [long_dialogue]
    assert embeddings.queries == [long_dialogue]
