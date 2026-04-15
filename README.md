# 🤖 RAG-бот для интернет-магазина автозапчастей

## 📌 Описание проекта

В рамках проекта был разработан RAG-бот (Retrieval-Augmented Generation) для интернет-магазина автозапчастей.  
Бот предназначен для автоматического ответа на часто задаваемые вопросы клиентов, а также для помощи менеджерам.

Система использует векторную базу данных (ChromaDB) для поиска релевантных фрагментов из базы знаний и генерации ответов с помощью LLM (OpenAI).

---

## 🎯 Основные возможности

Бот отвечает на вопросы:

- график работы магазина  
- сроки и способы доставки  
- способы оплаты  
- возврат товара  
- статус заказа  
- подбор автозапчастей  
- инструкции для менеджеров  

---

## 🧠 Как работает система (RAG)

1. Документы из папки `data/` разбиваются на чанки  
2. Каждый чанк преобразуется в embedding (вектор)  
3. Векторы сохраняются в ChromaDB  
4. При запросе пользователя:
   - создаётся embedding запроса
   - ищутся похожие чанки (retrieval)
   - формируется контекст
   - LLM генерирует ответ на основе контекста  

---

## 📂 Структура проекта

```
assistant_api/
│
├── data/
│   ├── faq.txt
│   ├── delivery.txt
│   ├── returns.txt
│   ├── support.txt
│   └── staff.txt
│
├── chroma_db/
├── rag_pipeline.py
├── vector_store.py
├── evaluate_ragas.py
└── README.md
```

---

## ⚙️ Установка и запуск

### Python 3.11.9

```
python -m venv venv311
venv311\Scripts\activate
```

```
pip install chromadb openai tiktoken numpy pandas datasets ragas langchain langchain-openai python-dotenv
```

Создать `.env`:

```
OPENAI_API_KEY=your_api_key
```

---

## 📥 Индексация

```
python -c "from rag_pipeline import RAGPipeline; p=RAGPipeline(collection_name='rag_collection', data_dir='data'); print(p.reindex(recreate_collection=True))"
```

---

## 📊 Оценка

```
python evaluate_ragas.py
```

## 💡 Выводы

- Один чанк = один вопрос  
- Важно совпадение формулировок  
- Retrieval важнее модели  

---

## 🚀 Улучшения

- reranking  
- better embeddings  
- hybrid search  

---

## 📎 Технологии

Python, OpenAI, ChromaDB, RAGAS, LangChain



