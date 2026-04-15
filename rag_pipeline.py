"""
Основной RAG pipeline для API режима.
Управляет потоком: запрос -> кеш -> vector search -> LLM -> ответ -> кеш.

Работает с несколькими txt-файлами из директории data.
"""

from typing import Dict, Any, List
import os
from openai import OpenAI

from vector_store import VectorStore
from cache import RAGCache


class RAGPipeline:
    """Основной pipeline для RAG системы в API режиме."""

    def __init__(
        self,
        collection_name: str = "rag_collection",
        cache_db_path: str = "rag_cache.db",
        data_dir: str = "data",
        model: str = "gpt-4o-mini",
        top_k: int = 5,
    ):
        """
        Инициализация RAG pipeline.

        Args:
            collection_name: имя коллекции в ChromaDB
            cache_db_path: путь к базе данных кеша
            data_dir: путь к директории с txt-файлами
            model: модель OpenAI для генерации ответов
            top_k: сколько документов брать из retrieval
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY не установлен")

        self.model = model
        self.data_dir = data_dir
        self.top_k = top_k
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        print("Инициализация векторного хранилища...")
        self.vector_store = VectorStore(collection_name=collection_name)

        # Если коллекция пустая — загружаем документы из директории
        if self.vector_store.collection.count() == 0:
            print(f"Загрузка документов из директории {data_dir}...")
            stats = self.vector_store.load_documents_from_dir(data_dir, recreate=False)
            print(f"Загрузка завершена: {stats}")

        print("Инициализация кеша...")
        self.cache = RAGCache(db_path=cache_db_path)

        print("RAG Pipeline инициализирован (API mode)")

    def reindex(self, recreate_collection: bool = True) -> Dict[str, Any]:
        """
        Переиндексация базы знаний.
        """
        print(f"Переиндексация документов из {self.data_dir}...")
        return self.vector_store.load_documents_from_dir(
            self.data_dir,
            recreate=recreate_collection,
        )

    def _create_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Создание промпта для LLM с контекстом.

        Args:
            query: вопрос пользователя
            context_docs: релевантные документы из векторного хранилища

        Returns:
            сформированный промпт
        """
        context_parts = []

        for i, doc in enumerate(context_docs, 1):
            metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            source = (
                doc.get("source", metadata.get("source", "unknown"))
                if isinstance(doc, dict)
                else "unknown"
            )
            text = doc.get("text", "") if isinstance(doc, dict) else str(doc)

            context_parts.append(
                f"Документ {i} (источник: {source}):\n{text}\n"
            )

        context = "\n".join(context_parts)

        prompt = f"""Ты — AI-ассистент интернет-магазина.
Ответь на вопрос пользователя строго на основе предоставленного контекста.

Контекст:
{context}

Вопрос: {query}

Инструкции:
- Отвечай только на основе предоставленного контекста
- Если в контексте нет точной информации, так и скажи
- Будь точным, кратким и понятным
- Отвечай на русском языке
- Если возможно, укажи источник

Ответ:"""

        return prompt

    def _generate_answer(self, prompt: str) -> str:
        """
        Генерация ответа через OpenAI API.

        Args:
            prompt: промпт для модели

        Returns:
            сгенерированный ответ
        """
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — полезный AI-ассистент интернет-магазина. "
                        "Отвечай только на основе предоставленного контекста "
                        "и не придумывай факты."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=500,
        )

        return response.choices[0].message.content.strip()

    def query(self, user_query: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Основной метод обработки запроса пользователя через API.

        Поток:
        1. Проверка кеша
        2. Если в кеше нет - поиск в векторном хранилище
        3. Формирование промпта с контекстом
        4. Генерация ответа через LLM API
        5. Сохранение в кеш

        Args:
            user_query: запрос пользователя
            use_cache: использовать ли кеш

        Returns:
            словарь с ответом и метаданными
        """
        print(f"\n{'=' * 60}")
        print(f"Запрос: {user_query}")
        print(f"{'=' * 60}")

        # Шаг 1: Проверка кеша
        if use_cache:
            print("[*] Проверка кеша...")
            cached_result = self.cache.get(user_query)

            if cached_result:
                print("[+] Ответ найден в кеше")

                cached_context = cached_result.get("context", [])
                normalized_context = []

                for item in cached_context:
                    if isinstance(item, dict):
                        normalized_context.append(item)
                    else:
                        normalized_context.append(
                            {
                                "text": str(item),
                                "source": "cache",
                                "metadata": {"source": "cache"},
                            }
                        )

                return {
                    "query": user_query,
                    "answer": cached_result["answer"],
                    "from_cache": True,
                    "context_docs": normalized_context,
                    "cached_at": cached_result.get("created_at"),
                }

            print("[-] Ответ не найден в кеше")

        # Шаг 2: Поиск релевантных документов
        print("[*] Поиск релевантных документов через API...")
        context_docs = self.vector_store.search(user_query, top_k=self.top_k)
        print(f"[+] Найдено {len(context_docs)} релевантных документов")

        # Шаг 3: Формирование промпта
        print("[*] Формирование промпта...")
        prompt = self._create_prompt(user_query, context_docs)

        # Шаг 4: Генерация ответа через API
        print(f"[*] Генерация ответа через OpenAI API ({self.model})...")
        answer = self._generate_answer(prompt)
        print("[+] Ответ получен от API")

        # Шаг 5: Сохранение в кеш
        if use_cache:
            print("[*] Сохранение в кеш...")
            self.cache.set(user_query, answer, context_docs)
            print("[+] Сохранено в кеш")

        return {
            "query": user_query,
            "answer": answer,
            "from_cache": False,
            "context_docs": context_docs,
            "model": self.model,
            "mode": "API",
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики системы.

        Returns:
            словарь со статистикой
        """
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "model": self.model,
            "mode": "API",
            "data_dir": self.data_dir,
            "top_k": self.top_k,
        }


if __name__ == "__main__":
    import sys

    try:
        pipeline = RAGPipeline(data_dir="data")

        # При необходимости можно принудительно переиндексировать:
        # print(pipeline.reindex(recreate_collection=True))

        test_queries = [
            "Какой у вас график работы?",
            "Сколько занимает доставка по городу?",
            "Как оформить возврат?",
        ]

        for query in test_queries:
            result = pipeline.query(query, use_cache=False)
            print(f"\n{'=' * 60}")
            print(f"Вопрос: {result['query']}")
            print(f"Из кеша: {result['from_cache']}")
            print(f"Ответ: {result['answer']}")
            print("Контекст:")
            for doc in result["context_docs"]:
                print(f"- {doc.get('source')}: {doc.get('text', '')[:200]}")
            print(f"{'=' * 60}\n")

        stats = pipeline.get_stats()
        print("\nСтатистика системы:")
        print(stats)

    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)