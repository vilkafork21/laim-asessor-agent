"""
Агент-асессор для автоматической разметки данных.

Основные возможности:
- Структурирование и суммаризация инструкций по разметке
- Инициализация RAG для few-shot примеров
- Асинхронная пакетная обработка с rate limiting
- Поддержка нескольких асессоров с голосованием

Классы:
- Asessor: главный класс агента-асессора
"""

import json
import logging
import time

import pandas as pd
from agent.prompts import (
    EXAMPLES_SUMMARYZATION_PROMPT,
    INSTRUCTION_PROMPT,
    INSTRUCTION_SUMMARYZATION_PROMPT,
    REVIEW_PROMPT,
    SYSTEM_PROMPT,
)
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from pydantic import BaseModel

from agent.pydantic_output import create_simple_output_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from retriever.doc_loaders import load_documents_from_directory
from retriever.retriever import DomainRetriever, QuestionAnswerRetriever
from utils import blacklist_remover, process_with_rate_limit

# =====================================================
# CONSTANTS
# =====================================================

# Maximum retry attempts for LLM calls
MAX_RETRY_ATTEMPTS: int = 3

# Delay between retry attempts (seconds)
RETRY_DELAY_SECONDS: int = 5
LONG_RETRY_DELAY_SECONDS: int = 10

# Number of examples to retrieve from RAG
EXAMPLES_SIZE: int = 15

# Maximum length for text truncation
TRUNCATION_LENGTH: int = 150

# Дефектов в корзинах 5-12%, и в top-k похожих примеров попадает в лучшем случае
# один: судья не видит, как выглядит нарушение, и перестаёт его распознавать.
DEFECT_EXAMPLES_QUOTA: int = 3


def _lowest_allowed_values(values_set: dict[str, set]) -> dict[str, float]:
    """Минимум шкалы по каждому критерию; нечисловые шкалы пропускаются."""
    lowest: dict[str, float] = {}
    for column, values in values_set.items():
        try:
            numeric = [float(value) for value in values]
        except (TypeError, ValueError):
            continue
        if numeric:
            lowest[column] = min(numeric)
    return lowest


def _needs_review(answer: dict[str, object], lowest: dict[str, float]) -> bool:
    """Оценка на нижней границе шкалы — кандидат на перепроверку."""
    for column, floor in lowest.items():
        if column not in answer:
            continue
        try:
            if float(answer[column]) == floor:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _is_defect_example(example: dict, lowest: dict[str, float]) -> bool:
    """Пример с оценкой на нижней границе шкалы — образец нарушения."""
    try:
        answer = json.loads(example["answer"])
    except (TypeError, ValueError, KeyError):
        return False
    return isinstance(answer, dict) and _needs_review(answer, lowest)


def _defect_pool(
    retriever, examples: list[dict], query: str, quota: int = DEFECT_EXAMPLES_QUOTA
) -> list[dict]:
    """Похожие на запрос примеры нарушений; без индекса — список как есть."""
    if retriever is None:
        return examples
    return retriever.hybrid_search(query, k=quota)


def _ensure_defect_examples(
    found: list[dict],
    pool: list[dict],
    lowest: dict[str, float],
    quota: int = DEFECT_EXAMPLES_QUOTA,
) -> list[dict]:
    """Дополняет найденные примеры образцами с минимальной оценкой."""
    if not pool or not lowest:
        return found

    missing = quota - sum(1 for example in found if _is_defect_example(example, lowest))
    if missing <= 0:
        return found
    seen = {example["question"] for example in found}
    extra = [example for example in pool if example["question"] not in seen][:missing]
    return found + extra


def _serialize_llm_record(record: dict[str, object]) -> str:
    """Сериализует структурированный контекст в стабильный JSON для LLM."""
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


# =====================================================
# ASSESSOR CLASS
# =====================================================


class Asessor:
    """
    Агент асессор для динамической разметки данных на мониторинге
    """

    def __init__(
        self,
        llm: GigaChat,
        embedding_model: GigaChatEmbeddings,
        dataset: pd.DataFrame,
        domain_rag_path: str = None,
        instruction: str = "",
        context_columns: list[str] = [],
        answer_columns: list[str] = [],
        instruction_summarization: bool = True,
        instruction_structuring: bool = True,
        examples_summarization: bool = False,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging initialized")

        self.llm = llm
        self.embedding_model = embedding_model
        self.dataset = dataset
        self.domain_rag_path = domain_rag_path

        self.instruction_structuring = instruction_structuring
        self.instruction_summarization = instruction_summarization
        if self.instruction_summarization:
            self.logger.info("СУММАРИЗАЦИЯ ИНСТРУКЦИИ ВКЛЮЧЕНА")
        else:
            self.logger.info("СУММАРИЗАЦИЯ ИНСТРУКЦИИ ОТКЛЮЧЕНА")

        self.examples_summarization = examples_summarization
        if self.examples_summarization:
            self.logger.info("СУММАРИЗАЦИЯ ПРИМЕРОВ ВКЛЮЧЕНА")
        else:
            self.logger.info("СУММАРИЗАЦИЯ ПРИМЕРОВ ОТКЛЮЧЕНА")

        self.INSTUCTION_SUMMARYZATION_PROMPT = INSTRUCTION_SUMMARYZATION_PROMPT
        self.EXAMPLES_SUMMARYZATION_PROMPT = EXAMPLES_SUMMARYZATION_PROMPT

        self.INSTRUCTION_PROMPT = INSTRUCTION_PROMPT
        # self.COLUMN_EXTRACTOR_PROMPT = COLUMN_EXTRACTOR_PROMPT_v3
        self.SYSTEM_PROMPT = SYSTEM_PROMPT

        self.context_columns = context_columns  # Столбцы с контекстом для асессора
        self.answer_columns = answer_columns  # Столбцы ответы асессора
        self.answer_columns_values_set = {}  # Возможные ответы асессора
        self.rag_documents: list[dict[str, dict[str, str]]] = []  # Документы RAG
        self._output_model: type[BaseModel] | None = None  # Pydantic модель для structured output

        self.examples_retriever: QuestionAnswerRetriever = None  # Retriever часть RAG
        self.examples_retriever_chain: RunnableLambda = (
            None  # Retriever + Augmnetation часть RAG
        )
        self.domain_retriever: DomainRetriever = None
        self.domain_retriever_chain: RunnableLambda = None

        self.instruction = instruction
        # Подготовка инструкции
        if self.instruction_structuring:
            self._make_structured_instruction()
            self.logger.info("INSTRUCTION STRUCTURING: SUCCESS")
        self.original_instruction = self.instruction
        # Суммаризация инструкции
        if self.instruction_summarization:
            try:
                self.instruction = self._summaryze(
                    ChatPromptTemplate.from_messages(
                        [self.INSTUCTION_SUMMARYZATION_PROMPT]
                    ),
                    self.instruction,
                )
            except Exception as e:
                self.logger.error(f"Ошибка при суммаризации: {e}. Выполняю retry")
                time.sleep(LONG_RETRY_DELAY_SECONDS)
                self.instruction = self._summaryze(
                    ChatPromptTemplate.from_messages(
                        [self.INSTUCTION_SUMMARYZATION_PROMPT]
                    ),
                    self.instruction,
                )
        # Выделение информативных колонок и критериев оценки
        # self._extract_dataset_columns()
        self._get_all_unique_answers()
        self.logger.debug("UNIQUE_ANSWERS EXTRACTED: SUCCESS")
        # Инициализация RAG для поиска релевантных запросу примеров.
        self._init_rag()
        self.logger.debug("EXAMPLES RETRIEVAL INITIALIZATION: SUCCESS")

        # Инициализация цепочки для подачи контекста в LLM и логирования
        system_prompt = ChatPromptTemplate.from_messages([self.SYSTEM_PROMPT])
        # Начальная цепочка для подачи контекста в LLM и логирования
        ## Пришел запрос -> нашли релевантные примеры из базы знаний -> подали на вход Гиге вместе с инструкцией
        self.printing_chain = self.retrieval_chain | system_prompt

        # Создаём Pydantic модель для structured output
        self._create_output_model()

        # Используем Pydantic structured output вместо JsonOutputParser
        self.agent_chain = self.printing_chain | self.llm.with_structured_output(
            self._output_model
        )
        # Второй проход: сниженные оценки перепроверяет «адвокат» с чистым контекстом —
        # без него судья штрафует за то, чего инструкция не запрещает.
        self.review_printing_chain = RunnableLambda(
            lambda user_input: {
                "instructions": self.instruction,
                "answer_columns_values_set": self.answer_columns_values_set,
                "user_input": user_input,
            }
        ) | ChatPromptTemplate.from_messages([REVIEW_PROMPT])
        self.review_chain = self.review_printing_chain | self.llm.with_structured_output(
            self._output_model
        )
        self.logger.debug("Asessor Agent chain: SUCCESS")

    def _create_output_model(self) -> None:
        """Создаёт Pydantic модель для structured output на основе dataset."""
        self._output_model = create_simple_output_model(
            answer_columns=self.answer_columns,
            dataset=self.dataset,
            model_name="AssessmentOutput",
        )
        self.logger.info(f"Создана Pydantic модель: {self._output_model.__name__}")

    def get_columns(self) -> dict[str, str]:
        return {
            "answer_columns": self.answer_columns,
            "context_columns": self.context_columns,
        }

    # =====================================================
    # INSTRUCTION PROCESSING
    # =====================================================

    def _make_structured_instruction(self) -> None:
        """
        При наличие инсструкции:
            Улучшение структурированности инструкции асессора при помощи добавления Markdown разметки после запроса к Гиге
        """
        if self.instruction != "":
            instruction_prompt = ChatPromptTemplate.from_messages(
                [self.INSTRUCTION_PROMPT]
            )
            instruction_chain = instruction_prompt | self.llm | StrOutputParser()
            for i in range(MAX_RETRY_ATTEMPTS):
                try:
                    self.instruction = instruction_chain.invoke(
                        {"instruction": self.instruction}
                    )
                    break
                except Exception as e:
                    logging.error(f"Ошибка при структуризаци инструкции: {e}")
                    time.sleep(RETRY_DELAY_SECONDS)
            self.logger.info(f"📝STRUCTURED INSTRUCTION📝\n {self.instruction}")
        else:
            self.logger.error(
                "Инструкция отсутствует, улучшение структуры не было выполнено"
            )

    def _summaryze(self, prompt: ChatPromptTemplate, text: str) -> str:
        """
        Метод суммаризации текста
        Входные параметры:
            prompt_template - ChatPromptTemplate суммаризации
            text - текст для суммаризации
        Результат - суммаризированный текст
        """
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"summarization_text": text})

    def _extract_dataset_columns(self) -> None:
        """
        Метод извлечения информативных для агента столбцов
        Извлекается 2 типа столбцов:
            1) Подаваемые на вход асессору. Например, запрос пользователя, промпт, прочая информация, необходимая для принятия решения
            2) Ответ асессора/ов. Оценки по N критериям
        """
        dataset_modes = self.dataset.mode()
        if len(dataset_modes):
            self.dataset = self.dataset.fillna(dataset_modes.iloc[0])
        examples_size = EXAMPLES_SIZE
        json_input_df_examples = (
            self.dataset.sample(frac=1).iloc[:examples_size].to_dict(orient="records")
        )
        for example in json_input_df_examples:
            for key, value in example.items():
                if type(value) is str:
                    example[key] = value[:TRUNCATION_LENGTH]
                else:
                    example[key] = value
        json_input_df_examples = {
            f"Пример {i}: ": example for i, example in enumerate(json_input_df_examples)
        }
        string_input_df_examples = json.dumps(
            json_input_df_examples, indent=4, ensure_ascii=False
        )

        cleaned_string_input_df_examples = blacklist_remover(string_input_df_examples)
        if self.examples_summarization:
            cleaned_string_input_df_examples = self._summaryze(
                ChatPromptTemplate.from_messages([self.EXAMPLES_SUMMARYZATION_PROMPT]),
                cleaned_string_input_df_examples,
            )

        self.logger.debug(
            f"EXAMPLE FOR EXTRACT DATASET COLUMNS STRING LENGHT: {len(cleaned_string_input_df_examples)}"
        )
        self.logger.debug(f"EXAMPLES LEN: {examples_size}")
        column_extractor_prompt = ChatPromptTemplate.from_messages(
            [self.COLUMN_EXTRACTOR_PROMPT]
        )
        column_extractor_chain = column_extractor_prompt | self.llm | JsonOutputParser()

        input_data = {
            "instructions": self.instruction,
            "columns": self.dataset.columns,
            "df_examples": cleaned_string_input_df_examples,
        }

        columns_extractor_input = (
            column_extractor_prompt.invoke(input_data).messages[0].content
        )
        self.logger.debug(f"👀COLUMNS EXTRACTOR INPUT👀\n{columns_extractor_input}")
        try:
            columns_json_output = column_extractor_chain.invoke(input_data)
        except Exception as e:
            self.logger.error(f"Ошибка при парсинге колонок: {e}. Выполняю retry")
            time.sleep(LONG_RETRY_DELAY_SECONDS)
            columns_json_output = column_extractor_chain.invoke(input_data)

        self.logger.info(
            f"Selected columns: \n{json.dumps(columns_json_output, indent=4, ensure_ascii=False)}"
        )

        self.context_columns = columns_json_output["columns_for_context"]
        self.answer_columns = columns_json_output["columns_for_answer"]

    def _get_all_unique_answers(self) -> None:
        """
        Получение уникальных значений столбцов с ответами асессора.
        *Список возможных ответов
        """
        answer_columns_values_set = {}
        for answer_column in self.answer_columns:
            answer_columns_values_set[answer_column] = set(
                self.dataset[answer_column].unique()
            )
        self.answer_columns_values_set = answer_columns_values_set

        self.logger.info(
            f"Возможные ответы агента асессора:\n{self.answer_columns_values_set}"
        )

    # =====================================================
    # RAG INITIALIZATION
    # =====================================================

    def _init_examples_rag(self):
        questions = [
            _serialize_llm_record(question)
            for question in self.dataset[self.context_columns].to_dict(orient="records")
        ]
        answers = [
            _serialize_llm_record(answer)
            for answer in self.dataset[self.answer_columns].to_dict(orient="records")
        ]
        self.retrieval_examples = [
            {"question": questions[i], "answer": answers[i]}
            for i in range(self.dataset.shape[0])
        ]

        self.examples_retriever = QuestionAnswerRetriever(
            embedding_model=self.embedding_model,
            examples=[
                {"question": ex["question"], "answer": ex["answer"]}
                for ex in self.retrieval_examples
            ],
        )
        self._lowest_values = _lowest_allowed_values(self.answer_columns_values_set)
        self.defect_examples = [
            example
            for example in self.retrieval_examples
            if _is_defect_example(example, self._lowest_values)
        ]
        self.logger.info(f"Примеров с минимальной оценкой в базе: {len(self.defect_examples)}")
        # Отдельный индекс по нарушениям: иначе в few-shot попадают случайные
        # дефекты, не связанные с оцениваемым запросом.
        self.defect_retriever = (
            QuestionAnswerRetriever(
                embedding_model=self.embedding_model,
                examples=self.defect_examples,
            )
            if self.defect_examples
            else None
        )

    def update_rag(self, new_dataset: pd.DataFrame):
        self.dataset = new_dataset
        self._init_examples_rag()
        logging.info(
            "Retrieval успешно обновлен на полную корзину с первичной валидации"
        )

    def _init_domain_rag(self):
        domain_documents = load_documents_from_directory(self.domain_rag_path)
        if not domain_documents:
            # FAISS.from_documents([]) падает IndexError: пустой домен — не ошибка
            self.logger.warning(
                "Domain RAG: 0 документов после разбора, доменный контекст отключён"
            )
            return
        self.domain_retriever = DomainRetriever(
            embedding_model=self.embedding_model, documents=domain_documents
        )

    def _init_rag(self):
        """
        Инициализация RAG на основе полученного датасета
        """
        self._init_examples_rag()
        if self.domain_rag_path is not None:
            self._init_domain_rag()
            self.logger.debug("DOMAIN RETRIEVAL INITIALIZATION: SUCCESS")
        self.retrieval_chain = RunnableLambda(
            lambda user_input: {
                "examples": [
                    "\n\nКонтекст оценки: " + val["question"] + "\nОценки: " + val["answer"]
                    for val in _ensure_defect_examples(
                        self.examples_retriever.hybrid_search(query=user_input),
                        _defect_pool(
                            self.defect_retriever, self.defect_examples, user_input
                        ),
                        self._lowest_values,
                    )
                ],
                "domain_knowledge": "\n".join(
                    [
                        res["content"]
                        for res in self.domain_retriever.hybrid_search(query=user_input)
                    ]
                )
                if self.domain_retriever
                else "Ориентируйся на собственные знания",
                "user_input": user_input,
                "instructions": self.instruction,
                "answer_columns_values_set": self.answer_columns_values_set,
            }
        )

    # =====================================================
    # MAIN EXECUTION
    # =====================================================

    async def run(self, input_df: pd.DataFrame, delay_seconds: int = 3) -> pd.DataFrame:
        """
        Запуск агента асессора
        Args:
            input_df: pd.DataFrame. Датафрейм с входными данными для асессора
        """
        if not all(column in input_df.columns for column in self.context_columns):
            raise ValueError(
                f"В датасете отсутствуют необходимые для асессора столбцы.\nСтолбцы, необходимые для асесссора:\n{self.context_columns}"
            )
        self.logger.info("Столбцы датасета соответствуют ожидаемым")

        batch_inputs = input_df[self.context_columns].to_dict(orient="records")
        batch_inputs = [_serialize_llm_record(user_input) for user_input in batch_inputs]
        self.logger.info(f"Всего трейсов: {len(batch_inputs)}")
        self.logger.debug(batch_inputs)

        results = await process_with_rate_limit(
            self.agent_chain,
            batch_inputs,
            delay_seconds=delay_seconds,
        )

        return await self._review_lowest(batch_inputs, results, delay_seconds)

    async def _review_lowest(
        self, batch_inputs: list[str], results: list, delay_seconds: int
    ) -> list:
        """Перепроверяет оценки на нижней границе шкалы вторым проходом."""
        lowest = _lowest_allowed_values(self.answer_columns_values_set)
        if not lowest:
            return results
        positions = [
            index
            for index, value in enumerate(results)
            if value is not None
            and _needs_review(
                value.model_dump() if hasattr(value, "model_dump") else value, lowest
            )
        ]
        if not positions:
            return results
        self.logger.info(f"Перепроверка сниженных оценок: {len(positions)} из {len(results)}")
        reviewed = await process_with_rate_limit(
            self.review_chain,
            [batch_inputs[index] for index in positions],
            delay_seconds=delay_seconds,
        )
        for index, verdict in zip(positions, reviewed):
            if verdict is not None:
                results[index] = verdict
        return results
