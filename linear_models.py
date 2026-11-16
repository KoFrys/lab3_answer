"""Лабораторная №3: линейные модели, написанные вручную на NumPy.

Содержит:
- train_linear_regression - линейная регрессия, пакетный градиентный спуск по MSE;
- train_logistic_classifier - бинарная логистическая регрессия, пакетный
  градиентный спуск по бинарной кросс-энтропии;
- compare_classical_models - сравнение с готовыми моделями scikit-learn на том
  же разбиении.

Внутри двух собственных моделей LinearRegression и LogisticRegression из
sklearn не используются.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

RANDOM_STATE = 42  # для воспроизводимости sklearn-моделей


# --------------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------------
def _as_matrix(features) -> np.ndarray:
    """Приводит признаки к числовой матрице (n_samples, n_features).

    Если пришёл DataFrame с текстовыми столбцами, значения Yes/No заменяются
    на 1/0, остальные категории кодируются номерами.
    """
    if isinstance(features, pd.DataFrame):
        frame = features.copy()
        for column in frame.columns:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                values = frame[column].astype(str).str.strip().str.lower()
                if set(values.unique()) <= {"yes", "no"}:
                    frame[column] = (values == "yes").astype(float)
                else:
                    frame[column] = pd.factorize(values)[0].astype(float)
        matrix = frame.to_numpy(dtype=float)
    else:
        matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return matrix


def _as_vector(target) -> np.ndarray:
    """Приводит целевую переменную к одномерному массиву float."""
    return np.asarray(target, dtype=float).ravel()


def _fit_scaler(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Среднее и стандартное отклонение по обучающей выборке."""
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0  # константный столбец не делим на ноль
    return mean, std


def _to_original_scale(
    weights: np.ndarray, bias: float, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, float]:
    """Переводит параметры из стандартизованных признаков в исходные.

    z = (x - mean) / std,  w_z * z + b_z = (w_z / std) * x + (b_z - sum(w_z * mean / std))
    """
    original_weights = weights / std
    original_bias = float(bias - np.sum(weights * mean / std))
    return original_weights, original_bias


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)  # защита от переполнения exp
    return 1.0 / (1.0 + np.exp(-z))


def _binary_cross_entropy(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _gd_linear(
    x: np.ndarray, y: np.ndarray, learning_rate: float, epochs: int
) -> tuple[np.ndarray, float, list[float]]:
    """Пакетный градиентный спуск для MSE. Возвращает w, b и историю MSE."""
    n_samples, n_features = x.shape
    weights = np.zeros(n_features)
    bias = 0.0
    history: list[float] = []
    for _ in range(epochs):
        error = x @ weights + bias - y
        history.append(float(np.mean(error ** 2)))
        weights = weights - learning_rate * (2.0 / n_samples) * (x.T @ error)
        bias = bias - learning_rate * 2.0 * float(np.mean(error))
    return weights, float(bias), history


def _gd_logistic(
    x: np.ndarray, y: np.ndarray, learning_rate: float, epochs: int
) -> tuple[np.ndarray, float, list[float]]:
    """Пакетный градиентный спуск для бинарной кросс-энтропии."""
    n_samples, n_features = x.shape
    weights = np.zeros(n_features)
    bias = 0.0
    history: list[float] = []
    for _ in range(epochs):
        probabilities = _sigmoid(x @ weights + bias)
        history.append(_binary_cross_entropy(y, probabilities))
        error = probabilities - y
        weights = weights - learning_rate * (x.T @ error) / n_samples
        bias = bias - learning_rate * float(np.mean(error))
    return weights, float(bias), history


def _binary_labels(target) -> np.ndarray:
    """Целевая переменная в виде 0/1 (если метки другие, большая метка = 1)."""
    y = _as_vector(target)
    classes = np.unique(y)
    if len(classes) == 2 and not set(classes) <= {0.0, 1.0}:
        y = (y == classes[1]).astype(float)
    return y


def classification_metrics(y_true, probabilities, predictions) -> ClassificationMetrics:
    """ROC-AUC, Accuracy, Precision, Recall и F1 на тестовой выборке."""
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
    )


# --------------------------------------------------------------------------
# Задания
# --------------------------------------------------------------------------
def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test.

    Признаки стандартизуются по обучающей выборке, затем выполняется пакетный
    градиентный спуск по MSE. Возвращаемые веса и смещение пересчитаны в шкалу
    исходных признаков, поэтому прогноз = X @ weights + bias для необработанных X.
    """
    x_train = _as_matrix(data.train_features)
    y_train = _as_vector(data.train_target)
    x_test = _as_matrix(data.test_features)

    mean, std = _fit_scaler(x_train)
    x_train_scaled = (x_train - mean) / std

    weights, bias, history = _gd_linear(
        x_train_scaled, y_train, data.learning_rate, data.epochs
    )
    weights, bias = _to_original_scale(weights, bias, mean, std)

    return RegressionResult(
        weights=weights,
        bias=bias,
        loss_history=history,
        predictions=x_test @ weights + bias,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test.

    Признаки стандартизуются по обучающей выборке, градиентный спуск минимизирует
    бинарную кросс-энтропию. Порог классификации - 0.5. Веса пересчитаны в шкалу
    исходных признаков.
    """
    x_train = _as_matrix(data.train_features)
    y_train = _binary_labels(data.train_target)
    x_test = _as_matrix(data.test_features)

    mean, std = _fit_scaler(x_train)
    x_train_scaled = (x_train - mean) / std

    weights, bias, history = _gd_logistic(
        x_train_scaled, y_train, data.learning_rate, data.epochs
    )
    weights, bias = _to_original_scale(weights, bias, mean, std)

    probabilities = _sigmoid(x_test @ weights + bias)
    return ClassificationResult(
        weights=weights,
        bias=bias,
        loss_history=history,
        probabilities=probabilities,
        predictions=(probabilities >= 0.5).astype(int),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    Модели обучаются на train и оцениваются на test из `data` (то же разбиение,
    что и у собственной модели). Для логистической регрессии и KNN признаки
    стандартизуются (чувствительны к масштабу), для дерева это не нужно.

    После запуска `python run_experiments.py` реальные результаты записываются в
    experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    x_train = _as_matrix(data.train_features)
    y_train = _binary_labels(data.train_target)
    x_test = _as_matrix(data.test_features)
    y_test = _binary_labels(data.test_target)

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }
    metrics = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = model.predict(x_test)
        metrics[name] = classification_metrics(y_test, probabilities, predictions)

    return BaselineComparisonResult(
        logistic_regression=metrics["logistic_regression"],
        decision_tree=metrics["decision_tree"],
        knn=metrics["knn"],
    )
