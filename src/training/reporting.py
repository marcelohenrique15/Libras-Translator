"""Gráficos e tabelas de uma avaliação, sem depender da arquitetura do modelo."""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


class ExperimentReporter:
    def __init__(self, output_dir: Path, class_labels: list[str]) -> None:
        self.output_dir = Path(output_dir)
        self.class_labels = list(class_labels)
        if not self.class_labels:
            raise ValueError("Informe os nomes das classes na ordem dos índices do modelo.")

    def write(
        self,
        history: list[dict],
        test_result: dict,
        search_report: dict | None = None,
        validation_histories: list[list[dict]] | None = None,
    ) -> None:
        metrics = test_result["metrics"]
        matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
        class_count = len(self.class_labels)
        if matrix.shape != (class_count, class_count) or np.any(matrix < 0):
            raise ValueError("A matriz de confusão deve ter uma linha e coluna por classe, com contagens >= 0.")
        predictions = self._prediction_rows(test_result)
        if int(matrix.sum()) != len(predictions):
            raise ValueError("A matriz de confusão e as previsões devem representar as mesmas amostras.")
        per_class = self._class_metrics(matrix)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "graphs").mkdir(exist_ok=True)
        self._write_csv("predictions.csv", predictions)
        self._write_csv("per_class.csv", per_class)
        self._plot_training(history)
        self._plot_confusion(matrix, normalized=False)
        self._plot_confusion(matrix, normalized=True)
        self._plot_per_class(per_class)
        if search_report is not None:
            self._plot_search(search_report)
        if validation_histories:
            self._plot_validation(validation_histories, "loss", "Perda")
            self._plot_validation(validation_histories, "f1_macro", "F1 macro")

    def _prediction_rows(self, result: dict) -> list[dict]:
        metrics = result["metrics"]
        expected, predicted = metrics["expected"], metrics["predicted"]
        sample_ids = result["sample_ids"]
        if not (len(sample_ids) == len(expected) == len(predicted)):
            raise ValueError("Identificadores, classes esperadas e previsões devem ter o mesmo tamanho.")
        probabilities = metrics.get("probabilities")
        if probabilities is not None:
            probabilities = np.asarray(probabilities, dtype=np.float64)
            if probabilities.shape != (len(sample_ids), len(self.class_labels)):
                raise ValueError("As probabilidades devem ter uma linha por amostra e uma coluna por classe.")
            if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
                raise ValueError("As probabilidades devem ser finitas e estar entre 0 e 1.")
            if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
                raise ValueError("As probabilidades de cada amostra devem somar 1.")

        rows = []
        for index, (sample_id, actual, prediction) in enumerate(zip(sample_ids, expected, predicted)):
            if not (0 <= actual < len(self.class_labels) and 0 <= prediction < len(self.class_labels)):
                raise ValueError("Uma previsão contém um índice de classe inválido.")
            rows.append({
                "sample_id": sample_id,
                "expected_index": actual,
                "expected_class": self.class_labels[actual],
                "predicted_index": prediction,
                "predicted_class": self.class_labels[prediction],
                "correct": actual == prediction,
                "confidence": float(probabilities[index, prediction]) if probabilities is not None else "",
            })
        return rows

    def _class_metrics(self, matrix: np.ndarray) -> list[dict]:
        support = matrix.sum(axis=1)
        predicted_count = matrix.sum(axis=0)
        correct = matrix.diagonal()
        precision = np.divide(correct, predicted_count, out=np.zeros(len(matrix)), where=predicted_count != 0)
        recall = np.divide(correct, support, out=np.zeros(len(matrix)), where=support != 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros(len(matrix)), where=(precision + recall) != 0)
        return [
            {
                "class_index": index, "class_label": label,
                "support": int(support[index]), "predicted_count": int(predicted_count[index]),
                "precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]),
            }
            for index, label in enumerate(self.class_labels)
        ]

    @staticmethod
    def _normalized_confusion(matrix: np.ndarray) -> np.ndarray:
        support = matrix.sum(axis=1, keepdims=True)
        return np.divide(matrix, support, out=np.zeros_like(matrix, dtype=np.float64), where=support != 0)

    def _write_csv(self, name: str, rows: list[dict]) -> None:
        if not rows:
            return
        with (self.output_dir / name).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _plot_training(self, history: list[dict]) -> None:
        if not history:
            return
        epochs = [row["epoch"] for row in history]
        for metric, ylabel, filename in (("loss", "Perda", "loss.png"), ("f1_macro", "F1 macro", "f1.png")):
            figure = Figure(figsize=(9, 5), layout="constrained")
            axis = figure.subplots()
            axis.plot(epochs, [row[f"train_{metric}"] for row in history], marker="o", label="Treino")
            if metric == "loss" and all("train_objective" in row for row in history):
                axis.plot(epochs, [row["train_objective"] for row in history], linestyle="--",
                          label="Objetivo de treino com regularização")
                axis.lines[0].set_label("Perda de classificação")
            self._format_curve(axis, "Treinamento final · duração fixada pela busca", ylabel)
            axis.legend()
            self._save(figure, filename)

    def _plot_validation(self, histories: list[list[dict]], metric: str, ylabel: str) -> None:
        histories = [history for history in histories if history]
        if not histories:
            return
        columns = min(3, len(histories))
        rows = (len(histories) + columns - 1) // columns
        figure = Figure(figsize=(5 * columns, 3.6 * rows), layout="constrained")
        axes = figure.subplots(rows, columns, squeeze=False)
        figure.suptitle(f"Busca · curvas de {ylabel.lower()} nas divisões de validação")
        for index, (axis, history) in enumerate(zip(axes.flat, histories)):
            epochs = [row["epoch"] for row in history]
            axis.plot(epochs, [row[f"train_{metric}"] for row in history], label="Treino")
            axis.plot(epochs, [row[f"validation_{metric}"] for row in history], label="Validação")
            self._format_curve(axis, f"Divisão {index + 1}", ylabel)
            axis.legend(fontsize=8)
        for axis in list(axes.flat)[len(histories):]:
            axis.set_visible(False)
        self._save(figure, f"validation_{metric}.png")

    @staticmethod
    def _format_curve(axis, title: str, ylabel: str) -> None:
        axis.set(title=title, xlabel="Época", ylabel=ylabel)
        axis.grid(alpha=0.25)
        if ylabel == "F1 macro":
            axis.set_ylim(0, 1.02)

    def _plot_confusion(self, matrix: np.ndarray, normalized: bool) -> None:
        values = self._normalized_confusion(matrix) if normalized else matrix
        size = max(7, len(self.class_labels) * 0.55)
        figure = Figure(figsize=(size + 2, size), layout="constrained")
        axis = figure.subplots()
        image = axis.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
        figure.colorbar(image, ax=axis, label="Proporção por classe real" if normalized else "Amostras")
        positions = np.arange(len(self.class_labels))
        axis.set_xticks(positions, self.class_labels, rotation=55, ha="right", fontsize=9)
        axis.set_yticks(positions, self.class_labels, fontsize=9)
        title = "Teste · matriz de confusão"
        axis.set(title=f"{title} · normalizada por classe real" if normalized else title,
                 xlabel="Classe prevista", ylabel="Classe real")
        maximum = float(values.max())
        for row in positions:
            for column in positions:
                value = values[row, column]
                label = f"{value:.0%}" if normalized else str(int(value))
                axis.text(column, row, label, ha="center", va="center", fontsize=8,
                          color="white" if maximum > 0 and value > maximum * 0.55 else "black")
        filename = "confusion_matrix_normalized.png" if normalized else "confusion_matrix.png"
        self._save(figure, filename)

    def _plot_per_class(self, metrics: list[dict]) -> None:
        figure = Figure(figsize=(11, max(5, len(metrics) * 0.48)), layout="constrained")
        axis = figure.subplots()
        positions = np.arange(len(metrics))
        for offset, metric, label in ((-0.25, "precision", "Precisão"), (0, "recall", "Recall"), (0.25, "f1", "F1")):
            axis.barh(positions + offset, [row[metric] for row in metrics], height=0.23, label=label)
        axis.set_yticks(positions, [f"{row['class_label']} (n={row['support']})" for row in metrics], fontsize=9)
        axis.invert_yaxis()
        axis.set(title="Teste · métricas por classe", xlabel="Pontuação", xlim=(0, 1.05))
        axis.grid(axis="x", alpha=0.25)
        axis.legend(loc="lower right")
        self._save(figure, "per_class.png")

    def _plot_search(self, report: dict) -> None:
        trials = [trial for trial in report["trials"]
                  if trial.get("state") == "COMPLETE" and trial.get("mean_validation_f1_macro") is not None]
        if not trials:
            return
        trials.sort(key=lambda trial: trial["number"])
        numbers = [trial["number"] + 1 for trial in trials]
        scores = [trial["mean_validation_f1_macro"] for trial in trials]
        figure = Figure(figsize=(10, 5), layout="constrained")
        axis = figure.subplots()
        axis.scatter(numbers, scores, label="Média das validações")
        axis.plot(numbers, np.maximum.accumulate(scores), label="Melhor média até a tentativa", color="tab:orange")
        for number, score in zip(numbers, scores):
            if number == report["best_trial"] + 1:
                axis.scatter(number, score, marker="*", s=200, color="tab:red", label="Configuração escolhida", zorder=3)
        axis.set(title="Optuna · seleção por F1 macro médio de validação", xlabel="Tentativa", ylabel="F1 macro médio")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend()
        self._save(figure, "search.png")

    def _save(self, figure: Figure, name: str) -> None:
        # Não registra janelas ou figuras no pyplot; usa somente o renderizador Agg.
        FigureCanvasAgg(figure)
        try:
            figure.savefig(self.output_dir / "graphs" / name, dpi=160)
        finally:
            figure.clear()
