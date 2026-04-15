"""
Модуль работы с векторным хранилищем ChromaDB.
Загружает несколько txt-файлов из директории data,
парсит CHUNK-структуру и выполняет поиск по векторам.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from openai import OpenAI
from dotenv import load_dotenv


env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


class VectorStore:
    """Векторное хранилище на основе ChromaDB."""

    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_directory: str = "./chroma_db",
    ):
        """
        Инициализация векторного хранилища.

        Args:
            collection_name: имя коллекции в ChromaDB
            persist_directory: директория для хранения данных
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY не установлен")

        self.client = chromadb.PersistentClient(path=persist_directory)

        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(
                f"Коллекция '{collection_name}' загружена. "
                f"Документов: {self.collection.count()}"
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"Создана новая коллекция '{collection_name}'")

        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def reset_collection(self) -> None:
        """
        Удаляет и пересоздаёт коллекцию.
        """
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Коллекция '{self.collection_name}' пересоздана")

    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Создание embeddings через OpenAI батчами.
        """
        embeddings: List[List[float]] = []
        batch_size = 100

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
            )

            embeddings.extend([item.embedding for item in response.data])
            print(f"Обработано {min(start + batch_size, len(texts))}/{len(texts)} чанков")

        return embeddings

    def _parse_chunked_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Парсит текст в формате:

        === CHUNK START ===
        ID: ...
        CATEGORY: ...
        TOPIC: ...
        QUESTION: ...
        ANSWER:
        ...
        KEYWORDS: ...
        === CHUNK END ===
        """
        pattern = r"=== CHUNK START ===(.*?)=== CHUNK END ==="
        matches = re.findall(pattern, text, re.DOTALL)

        parsed_chunks: List[Dict[str, Any]] = []

        for raw_chunk in matches:
            lines = [line.rstrip() for line in raw_chunk.strip().splitlines()]

            metadata: Dict[str, Any] = {}
            answer_lines: List[str] = []
            in_answer = False

            for raw_line in lines:
                line = raw_line.strip()

                if not line:
                    continue

                if line.startswith("ID:"):
                    metadata["chunk_id"] = line.replace("ID:", "", 1).strip()
                    in_answer = False

                elif line.startswith("CATEGORY:"):
                    metadata["category"] = line.replace("CATEGORY:", "", 1).strip()
                    in_answer = False

                elif line.startswith("TOPIC:"):
                    metadata["topic"] = line.replace("TOPIC:", "", 1).strip()
                    in_answer = False

                elif line.startswith("QUESTION:"):
                    metadata["question"] = line.replace("QUESTION:", "", 1).strip()
                    in_answer = False

                elif line.startswith("ANSWER:"):
                    in_answer = True

                elif line.startswith("KEYWORDS:"):
                    metadata["keywords"] = line.replace("KEYWORDS:", "", 1).strip()
                    in_answer = False

                elif in_answer:
                    answer_lines.append(line)

            question = metadata.get("question", "")
            answer = "\n".join(answer_lines).strip()

            if not question and not answer:
                continue

            # Формируем текст чанка в retrieval-friendly виде
            text_block = f"Вопрос: {question}\nОтвет: {answer}".strip()

            parsed_chunks.append(
                {
                    "text": text_block,
                    "metadata": metadata,
                }
            )

        return parsed_chunks

    def _read_txt_files_from_dir(self, data_dir: str) -> List[Dict[str, Any]]:
        """
        Читает все txt-файлы из директории и парсит чанки.
        """
        directory = Path(data_dir)

        if not directory.exists():
            raise FileNotFoundError(f"Директория не найдена: {data_dir}")

        txt_files = sorted(directory.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"В директории {data_dir} нет .txt файлов")

        all_chunks: List[Dict[str, Any]] = []

        for file_path in txt_files:
            print(f"Загрузка файла: {file_path.name}")

            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                continue

            parsed_chunks = self._parse_chunked_text(content)

            for idx, chunk in enumerate(parsed_chunks):
                metadata = chunk["metadata"]
                metadata["source"] = file_path.name
                metadata["chunk_index"] = idx

                # Уникальный id из файла + chunk_id или индекса
                chunk_id = metadata.get("chunk_id", f"{file_path.stem}_{idx}")
                chunk["id"] = f"{file_path.stem}_{chunk_id}"

            all_chunks.extend(parsed_chunks)

        return all_chunks

    def load_documents_from_dir(self, data_dir: str, recreate: bool = False) -> Dict[str, Any]:
        """
        Загружает документы из директории в ChromaDB.

        Args:
            data_dir: директория с txt-файлами
            recreate: пересоздать ли коллекцию перед загрузкой
        """
        if recreate:
            self.reset_collection()

        chunks = self._read_txt_files_from_dir(data_dir)

        if not chunks:
            raise ValueError("Не удалось сформировать чанки из директории")

        texts = [chunk["text"] for chunk in chunks]
        ids = [chunk["id"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        print(f"Всего чанков: {len(chunks)}")

        embeddings = self._create_embeddings(texts)

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        stats = {
            "collection_name": self.collection_name,
            "documents_loaded": len(chunks),
            "sources": sorted(list({m["source"] for m in metadatas})),
            "persist_directory": self.persist_directory,
        }

        print(f"Загружено {len(chunks)} документов в коллекцию '{self.collection_name}'")
        return stats

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов по запросу.

        Args:
            query: текст запроса
            top_k: количество документов для возврата
        """
        query_embedding = self._create_embeddings([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        docs = results.get("documents", [[]])[0] or []
        metas = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []

        formatted_results: List[Dict[str, Any]] = []

        for i, doc_text in enumerate(docs):
            raw_metadata = metas[i] if i < len(metas) else {}
            metadata = raw_metadata if isinstance(raw_metadata, dict) else {}

            distance = distances[i] if i < len(distances) else None

            formatted_results.append(
                {
                    "text": doc_text or "",
                    "metadata": metadata,
                    "source": metadata.get("source", "unknown"),
                    "score": distance,
                }
            )

        return formatted_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Получение статистики коллекции.
        """
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "persist_directory": self.persist_directory,
        }


if __name__ == "__main__":
    import sys

    try:
        vector_store = VectorStore(collection_name="test_collection")

        if Path("data").exists():
            print(vector_store.load_documents_from_dir("data", recreate=True))

        results = vector_store.search("Как подобрать автозапчасть?", top_k=3)

        print("\nРезультаты поиска:")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. SOURCE={doc['source']}")
            print(doc["text"][:300])
            print(f"score={doc['score']}")

        print("\nСтатистика:")
        print(vector_store.get_collection_stats())

    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)