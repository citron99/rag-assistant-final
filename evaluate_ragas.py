"""
Оценка качества RAG системы через RAGAS для assistant_api.

Использует:
- RAGPipeline для получения ответов
- RAGAS для оценки метрик:
    - Faithfulness
    - Context Precision
    - Answer Relevancy (если доступен embeddings backend)

Работает с базой знаний из нескольких txt-файлов в директории data.
"""

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# ---------------------------------
# Загрузка .env
# ---------------------------------
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from datasets import Dataset
from ragas import evaluate
from rag_pipeline import RAGPipeline

# ---------------------------------
# Импорт базовых метрик RAGAS
# ---------------------------------
faithfulness_metric = None
context_precision_metric = None

try:
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._context_precision import ContextPrecision

    faithfulness_metric = Faithfulness
    context_precision_metric = ContextPrecision
except ImportError:
    try:
        from ragas.metrics.collections import faithfulness, context_precision

        faithfulness_metric = faithfulness
        context_precision_metric = context_precision
    except ImportError:
        from ragas.metrics import faithfulness, context_precision

        faithfulness_metric = faithfulness
        context_precision_metric = context_precision

# ---------------------------------
# Импорт Answer Relevancy
# ---------------------------------
answer_relevancy_class = None

try:
    from ragas.metrics import AnswerRelevancy as answer_relevancy_class
except ImportError:
    try:
        from ragas.metrics import answer_relevancy as answer_relevancy_class
    except ImportError:
        answer_relevancy_class = None


# ---------------------------------
# Контрольные вопросы под интернет-магазин
# ---------------------------------
EVALUATION_CASES = [
    {
        "question": "Какой у вас график работы?",
        "ground_truth": (
            "Интернет-магазин принимает заказы круглосуточно. "
            "Менеджеры обрабатывают заказы с понедельника по пятницу с 9:00 до 18:00. "
            "Суббота и воскресенье — выходные."
        ),
    },
    {
        "question": "Сколько занимает доставка по городу?",
        "ground_truth": (
            "По городу доставка занимает от 1 до 3 рабочих дней "
            "после подтверждения заказа менеджером."
        ),
    },
    {
        "question": "Какие способы доставки доступны?",
        "ground_truth": (
            "Доступны курьерская доставка, доставка в пункт выдачи и самовывоз. "
            "Выбор зависит от города и доступности услуги."
        ),
    },
    {
        "question": "Можно ли изменить заказ после оформления?",
        "ground_truth": (
            "Да, заказ можно изменить до момента передачи в доставку. "
            "Для этого нужно связаться с менеджером и сообщить номер заказа."
        ),
    },
    {
        "question": "Можно ли отменить заказ?",
        "ground_truth": (
            "Да, отмена возможна до отправки товара. "
            "Если заказ уже передан в доставку, его нельзя отменить, "
            "но можно оформить возврат после получения."
        ),
    },
    {
        "question": "Есть ли самовывоз?",
        "ground_truth": (
            "Да, доступен самовывоз из пункта выдачи после подтверждения заказа менеджером. "
            "Заказ можно забрать после уведомления о готовности."
        ),
    },
    {
        "question": "Можно ли вернуть товар, если он не подошёл?",
        "ground_truth": (
            "Да, возврат возможен в течение 14 дней, если товар в оригинальном состоянии "
            "и не имеет следов использования."
        ),
    },
    {
        "question": "Кто оплачивает доставку при возврате?",
        "ground_truth": (
            "Если возврат по вине магазина, доставка оплачивается магазином. "
            "Если возврат по личной причине, доставку оплачивает клиент."
        ),
    },
    {
        "question": "Как оформить возврат?",
        "ground_truth": (
            "Необходимо связаться с поддержкой, указать номер заказа "
            "и получить инструкции от менеджера."
        ),
    },
    {
        "question": "Что должен сделать менеджер при обработке заказа?",
        "ground_truth": (
            "Менеджер должен проверить наличие товара, проверить контактные данные "
            "и подтвердить заказ."
        ),
    },
    {
        "question": "Что делать, если доставка задерживается?",
        "ground_truth": (
            "Нужно связаться с поддержкой и сообщить номер заказа. "
            "Менеджер проверит статус доставки."
        ),
    },
    {
        "question": "Можно ли выбрать время доставки?",
        "ground_truth": (
            "В некоторых городах можно выбрать интервал доставки, "
            "это уточняется при подтверждении заказа."
        ),
    },
]


def ensure_metric_object(metric_candidate):
    """
    Приводит метрику к инициализированному объекту.
    """
    if metric_candidate is None:
        return None

    if hasattr(metric_candidate, "name") and not isinstance(metric_candidate, type):
        return metric_candidate

    if callable(metric_candidate):
        try:
            return metric_candidate()
        except TypeError:
            return metric_candidate

    return metric_candidate


def build_ragas_embeddings():
    """
    Пытается создать embeddings backend для Answer Relevancy.
    """
    try:
        from ragas.embeddings import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    except Exception:
        pass

    try:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    except Exception:
        pass

    return None


def prepare_dataset(pipeline: RAGPipeline, cases: List[Dict[str, str]]) -> Dataset:
    """
    Подготовка Dataset для RAGAS.
    """
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []

    print("[*] Получение ответов от RAG системы...\n")

    for i, case in enumerate(cases, 1):
        question = case["question"]
        ground_truth = case["ground_truth"]

        print(f"  {i}/{len(cases)}: {question}")

        result = pipeline.query(question, use_cache=False)

        questions_list.append(question)
        answers_list.append(result.get("answer", ""))

        context_docs = result.get("context_docs", [])
        context_texts = []

        for doc in context_docs:
            if isinstance(doc, dict):
                context_texts.append(doc.get("text", ""))
            else:
                context_texts.append(str(doc))

        contexts_list.append(context_texts)
        ground_truths_list.append(ground_truth)

        print("     [+] Ответ получен от OpenAI API")

    print()

    return Dataset.from_dict(
        {
            "question": questions_list,
            "answer": answers_list,
            "contexts": contexts_list,
            "ground_truth": ground_truths_list,
        }
    )


def safe_metric_mean(values: List[Any]) -> float:
    """
    Безопасно считает среднее значение метрики, игнорируя None и NaN.
    """
    numeric_values = []

    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        numeric_values.append(v)

    return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0


def result_to_dict(result) -> Dict[str, List[Any]]:
    """
    Безопасно преобразует результат RAGAS в обычный словарь.
    """
    if isinstance(result, dict):
        return result

    if hasattr(result, "_scores_dict"):
        return result._scores_dict

    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        return {col: df[col].tolist() for col in df.columns}

    raise TypeError(f"Не удалось преобразовать result в dict. Тип: {type(result)}")


def print_summary_metrics(result, use_answer_relevancy: bool) -> None:
    """
    Печатает средние значения метрик.
    """
    result_dict = result_to_dict(result)

    avg_faithfulness = safe_metric_mean(result_dict.get("faithfulness", []))
    avg_context_precision = safe_metric_mean(result_dict.get("context_precision", []))

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)
    print()
    print("[МЕТРИКИ] Средние значения:")
    print(f"   Faithfulness (точность ответа):          {avg_faithfulness:.4f}")
    print(f"   Context Precision (точность контекста):  {avg_context_precision:.4f}")

    metric_values = [avg_faithfulness, avg_context_precision]

    if use_answer_relevancy and "answer_relevancy" in result_dict:
        avg_answer_relevancy = safe_metric_mean(result_dict.get("answer_relevancy", []))
        print(f"   Answer Relevancy (релевантность ответа): {avg_answer_relevancy:.4f}")
        metric_values.append(avg_answer_relevancy)

    avg_score = sum(metric_values) / len(metric_values)

    print(f"\n{'─' * 70}")
    print(f"[ИТОГО] Средний балл: {avg_score:.4f}")

    if avg_score >= 0.85:
        print("   Оценка: Отличное качество! [OK]")
        print("   Система показывает очень высокую точность и релевантность ответов.")
    elif avg_score >= 0.70:
        print("   Оценка: Хорошее качество [OK]")
        print("   Система работает уверенно, но её ещё можно улучшить.")
    elif avg_score >= 0.50:
        print("   Оценка: Удовлетворительное качество [!]")
        print("   Рекомендуется улучшить чанки, FAQ-файлы или retrieval.")
    else:
        print("   Оценка: Требует значительного улучшения [X]")
        print("   Нужно пересмотреть стратегию chunking, retrieval и качество данных.")


def print_detailed_results(result, cases: List[Dict[str, str]], use_answer_relevancy: bool) -> None:
    """
    Печатает результаты по каждому вопросу.
    """
    result_dict = result_to_dict(result)

    faithfulness_scores = result_dict.get("faithfulness", [])
    context_precision_scores = result_dict.get("context_precision", [])
    answer_relevancy_scores = result_dict.get("answer_relevancy", [])

    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ВОПРОСАМ")
    print("=" * 70)

    for i, case in enumerate(cases):
        question = case["question"]
        print(f"\n{i + 1}. {question}")

        faith_val = faithfulness_scores[i] if i < len(faithfulness_scores) else None
        if faith_val is not None and not (isinstance(faith_val, float) and math.isnan(faith_val)):
            print(f"   Faithfulness:       {faith_val:.4f}")
        else:
            print("   Faithfulness:       не удалось вычислить")

        cp_val = context_precision_scores[i] if i < len(context_precision_scores) else None
        if cp_val is not None and not (isinstance(cp_val, float) and math.isnan(cp_val)):
            print(f"   Context Precision:  {cp_val:.4f}")
        else:
            print("   Context Precision:  не удалось вычислить")

        if use_answer_relevancy and answer_relevancy_scores:
            ar_val = answer_relevancy_scores[i] if i < len(answer_relevancy_scores) else None
            if ar_val is not None and not (isinstance(ar_val, float) and math.isnan(ar_val)):
                print(f"   Answer Relevancy:   {ar_val:.4f}")
            else:
                print("   Answer Relevancy:   не удалось вычислить")


def evaluate_rag_system():
    """
    Основная функция оценки RAG-системы через RAGAS.
    """
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ (API MODE) ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()

    if not os.getenv("OPENAI_API_KEY"):
        print("[ОШИБКА] OPENAI_API_KEY не установлен")
        print("\nУстановите переменную окружения:")
        print("  Windows (PowerShell): $env:OPENAI_API_KEY='your-key'")
        print("  Windows (CMD): set OPENAI_API_KEY=your-key")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key'")
        print("\nИли создайте файл .env в корне проекта с содержимым:")
        print("  OPENAI_API_KEY=your-key-here")
        sys.exit(1)

    try:
        print("[*] Инициализация RAG системы (API mode)...\n")
        pipeline = RAGPipeline(
            collection_name="rag_collection",
            cache_db_path="api_rag_cache.db",
            data_dir="data",
            model="gpt-4o-mini",
            top_k=5,
        )
        print("\n[OK] RAG система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации RAG pipeline: {e}")
        sys.exit(1)

    print("=" * 70)
    dataset = prepare_dataset(pipeline, EVALUATION_CASES)
    print("=" * 70)

    print("\n[*] Подготовка метрик RAGAS...")

    metrics_to_use = []

    faithfulness_obj = ensure_metric_object(faithfulness_metric)
    context_precision_obj = ensure_metric_object(context_precision_metric)

    if faithfulness_obj is None or not hasattr(faithfulness_obj, "name"):
        print("[ОШИБКА] Faithfulness не инициализирована как объект метрики")
        sys.exit(1)

    if context_precision_obj is None or not hasattr(context_precision_obj, "name"):
        print("[ОШИБКА] Context Precision не инициализирована как объект метрики")
        sys.exit(1)

    metrics_to_use.append(faithfulness_obj)
    metrics_to_use.append(context_precision_obj)

    use_answer_relevancy = False
    evaluator_embeddings = None

    try:
        from ragas.metrics import AnswerRelevancy
        evaluator_embeddings = build_ragas_embeddings()

        if evaluator_embeddings is not None:
            ar_obj = AnswerRelevancy(
                embeddings=evaluator_embeddings,
                strictness=3,
            )

            if hasattr(ar_obj, "name"):
                metrics_to_use.append(ar_obj)
                use_answer_relevancy = True
                print("   [+] Answer Relevancy включена")
            else:
                print("   [!] Answer Relevancy не удалось привести к объекту метрики")
        else:
            print("   [!] Answer Relevancy отключена: не удалось создать embeddings backend")

    except Exception as e:
        print(f"   [!] Answer Relevancy отключена: {e}")

    print("\n[*] Запуск оценки метрик RAGAS...")
    if use_answer_relevancy:
        print("   Метрики: Faithfulness, Context Precision, Answer Relevancy")
    else:
        print("   Метрики: Faithfulness, Context Precision")
    print("   (это займёт 1-2 минуты, так как RAGAS использует OpenAI для оценки)\n")

    try:
        if use_answer_relevancy and evaluator_embeddings is not None:
            result = evaluate(
                dataset=dataset,
                metrics=metrics_to_use,
                embeddings=evaluator_embeddings,
            )
        else:
            result = evaluate(
                dataset=dataset,
                metrics=metrics_to_use,
            )
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        sys.exit(1)

    print_summary_metrics(result, use_answer_relevancy)
    print_detailed_results(result, EVALUATION_CASES, use_answer_relevancy)

    print("\n" + "=" * 70)
    print("[INFO] ПОЯСНЕНИЯ К МЕТРИКАМ")
    print("=" * 70)
    print("""
Faithfulness:
  Показывает, насколько ответ опирается на найденный контекст.

Context Precision:
  Показывает, насколько retrieval нашёл правильные фрагменты.

Answer Relevancy:
  Показывает, насколько ответ вообще отвечает на вопрос пользователя.
""")

    if not use_answer_relevancy:
        print("""
ПРИМЕЧАНИЕ:
  Answer Relevancy не была включена.
  Это не ломает оценку полностью, но делает её менее полной.
""")

    print("=" * 70)
    print("[OK] Оценка завершена!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()