"""
Зе бест функции собраны здесь для удобства
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    fbeta_score,
    precision_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer


def get_scores(y_true, y_pred) -> pd.Series:
    return pd.Series(
        np.array(
            [
                accuracy_score(y_true, y_pred),
                recall_score(y_true, y_pred),
                f1_score(y_true, y_pred),
                precision_score(y_true, y_pred),
                fbeta_score(y_true, y_pred, beta=1.25),
                roc_auc_score(y_true, y_pred),
            ]
        ),
        index=["Accuracy", "Recall", "F1", "Precision", "F_beta", "ROC_AUC"],
    )


def optimize_pipeline(df_features, target, param_grid, modelSignature, beta=1.25):

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, target, test_size=0.2, random_state=42, stratify=target
    )
    # Стратифицированное разделение данных на обучающую и тестовую выборки гарантирует,
    # что в каждой из выборок будет представлено одинаковое соотношение классов.

    # Создаем пайплайн
    pipeline = Pipeline(
        [
            ("preprocessor", TfidfVectorizer()),
            ("normalizer", Normalizer()),
            modelSignature,
        ]
    )

    # Определяем метрику f_beta
    scorer = make_scorer(fbeta_score, beta=beta)

    # Создаем кросс-валидатор
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Создаем GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scorer,
        n_jobs=-1,
        verbose=10,
        error_score="raise",
    )

    # Обучаем GridSearchCV
    grid_search.fit(X_train, y_train)

    # Проверяем лучшую модель на тестовой выборке
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_score = fbeta_score(y_test, y_pred, beta=beta)

    print("F_beta score on test set: ", test_score)
    print("best params: ", grid_search.best_params_)
    return best_model, get_scores(y_test, y_pred)


import re
from collections import defaultdict

keywords_python = {
    "def",
    "self",
    "yield",
    "global",
    "nonlocal",
    "except",
    "raise",
    "with",
    "del",
    "lambda",
    "async",
    "await",
    "pass",
    "finally",
    "from",
    "as",
    "elif",
    "True",
    "False",
    "None",
    "logging",
    "re",
    "pathlib",
    "platform",
    "typing",
    "urllib",
    "selenium",
    "forge",
}

keywords_go = {
    "goroutine",
    "channel",
    "slice",
    "fallthrough",
    "defer",
    "select",
    "package",
    "go",
    "interface",
    "range",
    "make",
    "cap",
    "len",
    "func",
    "map",
    "struct",
    "type",
    "var",
    "const",
    "protobuf",
    "descriptor",
    "Enum",
    "Message",
    "ProtoMessage",
    "reflect",
    "dynamicpb",
    "protodesc",
}

keywords_cpp = {
    "template",
    "virtual",
    "friend",
    "inline",
    "typedef",
    "using",
    "#include",
    "#define",
    "namespace",
    "public",
    "private",
    "protected",
    "constexpr",
    "nullptr",
    "throw",
    "BM_NestedForLoopParallelCols",
    "BM_NestedForLoopParallelRows",
    "BM_NestedForLoopParallelCollapse",
    "math::bn254::Fr",
}

keywords_java = {
    "overriding",
    "overloading",
    "polymorphism",
    "encapsulation",
    "abstraction",
    "serialization",
    "reflection",
    "threadlocal",
    "enumeration",
    "assertion",
    "strictfp",
    "synchronized",
    "transient",
    "volatile",
}

keywords_typescript = {
    "generic",
    "union",
    "intersection",
    "readonly",
    "never",
    "unknown",
    "infer",
    "symbol",
    "unique",
}


# Функция для удаления строк и комментариев
def remove_strings_and_comments(code):
    code = re.sub(r"//.*?(\n|$)", "", code)  # однострочные комментарии
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)  # многострочные комментарии
    code = re.sub(r'".*?(?<!\\)"', '""', code)  # строковые литералы
    code = re.sub(r"'.*?(?<!\\)'", "''", code)  # символы
    return code


def tokenize_code(code):
    # Токенизация кода
    return re.findall(r"\b\w+\b", code)


def determine_language(code):

    # Проверка на наличие уникальных синтаксических конструкций
    if re.match(r".*#include.*", code, re.DOTALL):
        return "C++"
    if re.match(r".*import java.*", code, re.DOTALL):
        return "Java"
    if re.match(r".*package.*import.*", code, re.DOTALL):
        return "Go"
    if re.match(r".*type.*|.*declare.*|.*export.*", code, re.DOTALL):
        return "TypeScript"
    if re.match(r".*def.*|.*self.*", code, re.DOTALL):
        return "Python"

    tokens = tokenize_code(code)
    scores = defaultdict(int)

    # Проверка на наличие специфичных для языка библиотек и модулей
    if (
        "numpy" in tokens
        or "pandas" in tokens
        or "scipy" in tokens
        or "matplotlib" in tokens
    ):
        scores["Python"] += 10
    if (
        "java.util" in tokens
        or "java.io" in tokens
        or "javax.swing" in tokens
        or "java.net" in tokens
    ):
        scores["Java"] += 10
    if (
        "@angular/core" in tokens
        or "rxjs" in tokens
        or "typescript" in tokens
        or "zone.js" in tokens
    ):
        scores["TypeScript"] += 10
    if (
        "fmt" in tokens
        or "net/http" in tokens
        or "os" in tokens
        or "io/ioutil" in tokens
    ):
        scores["Go"] += 10
    if (
        "iostream" in tokens
        or "vector" in tokens
        or "algorithm" in tokens
        or "string" in tokens
    ):
        scores["C++"] += 10
    # Анализ структуры кода
    if re.search(r"^\s+", code, re.MULTILINE):
        scores["Python"] += 5
    if re.search(r"\{.*\}", code, re.DOTALL):
        scores["C++"] += 5
        scores["Java"] += 5
        scores["Go"] += 5
        scores["TypeScript"] += 5
    if re.search(r"func\s+\w+\s*\(", code):
        scores["Go"] += 5
    if re.search(r"\btype\s+\w+\s*:", code):
        scores["TypeScript"] += 5

    code = remove_strings_and_comments(code)

    # Подсчет ключевых слов
    for token in tokens:
        if token in keywords_python:
            scores["Python"] += 1
        if token in keywords_go:
            scores["Go"] += 1
        if token in keywords_cpp:
            scores["C++"] += 1
        if token in keywords_java:
            scores["Java"] += 1
        if token in keywords_typescript:
            scores["TypeScript"] += 1

    if scores:
        return max(scores, key=scores.get)
    return "Unknown"
