import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)
def plot_feature_importance(feature_importance, figsize=(12, 8)):
    """
    Creates an enhanced feature importance visualization with more detailed labels.

    :param feature_importance: DataFrame with feature importance values
    :param figsize: Figure size
    """
    plt.figure(figsize=figsize)

    bars = plt.barh(
        feature_importance["feature"],
        feature_importance["importance"],
        color="skyblue",
        alpha=0.8,
    )

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.5f}",
            ha="left",
            va="center",
            fontsize=10,
        )

    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance Analysis", pad=20)

    # Add gridlines and invert y-axis for better readability
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.gca().invert_yaxis()

    # Customize the plot with a more descriptive title and axis labels
    plt.title("Feature Importance: Top Contributors to Model Predictions")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")

    plt.tight_layout()

def plot_confusion_matrix_analysis(y_true, y_pred, clf, figsize=(15, 5)):
    """
    Creates a comprehensive confusion matrix analysis with metrics and additional insights.

    :param y_true: True labels
    :param y_pred: Predicted labels
    :param clf: Classifier used for prediction
    :param figsize: Figure size
    """
    conf_matrix = confusion_matrix(y_true, y_pred)

    classes = clf.classes_
    class_names = [
        f"Class {c}" if isinstance(c, (int, np.integer)) else str(c) for c in classes
    ]

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1,
    )
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Confusion Matrix")

    metrics_data = pd.DataFrame(
        {"Precision": precision, "Recall": recall, "F1-Score": f1}, index=class_names
    )

    sns.heatmap(metrics_data, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax2)
    ax2.set_title("Performance Metrics by Class")

    plt.tight_layout()

def analyze_model_performance(feature_importance, y_true, y_pred, clf, save_dir="validation"):
    """
    Performs comprehensive model analysis, saving plots and results.

    :param feature_importance: DataFrame with feature importance values
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param clf: Classifier used for prediction
    :param save_dir: Directory to save plots and results (default: "validation")
    """

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    plot_feature_importance(feature_importance)
    feature_importance_filepath = os.path.join(save_dir, "feature_importance.png")
    plt.savefig(feature_importance_filepath)
    plt.close()

    plot_confusion_matrix_analysis(y_true, y_pred, clf)
    confusion_matrix_filepath = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_filepath)
    report = classification_report(y_true, y_pred)
    report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
    report_csv_filepath = os.path.join(save_dir, "classification_report.csv")
    report_df.to_csv(report_csv_filepath, index=True)
    plt.close()