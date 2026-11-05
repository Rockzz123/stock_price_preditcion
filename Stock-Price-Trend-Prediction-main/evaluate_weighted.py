import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from joblib import load
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
STOCKS = ['AAPL', 'AMZN', 'MSFT', 'NFLX']
MODEL_NAMES = ['rf', 'svm', 'lr', 'knn', 'mlp']
ENSEMBLE_NAME = 'WEIGHTED_ENSEMBLE'
FEATURE_COLUMNS = ['Tweet_Sentiment', 'News_Sentiment']

INPUT_DIR = 'Models'
OUTPUT_DIR = 'Results'


def save_confusion_matrix(cm, stock_ticker, model_name):
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Predicted Down', 'Predicted Up'],
            yticklabels=['Actual Down', 'Actual Up']
        )
        plt.title(f'Confusion Matrix for {stock_ticker} ({model_name})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f'{stock_ticker}_confusion_matrix_{model_name}.png'
            )
        )
        plt.close()
    except Exception as exc:
        print(f'Could not generate confusion matrix: {exc}')


def save_roc_curve_plot(roc_plot_data, stock_ticker):
    try:
        plt.figure(figsize=(10, 8))
        sorted_data = sorted(
            roc_plot_data.items(),
            key=lambda item: item[1][2],
            reverse=True
        )

        for model_name, (fpr, tpr, roc_auc) in sorted_data:
            plt.plot(
                fpr, tpr,
                label=f'{model_name} (AUC = {roc_auc:.3f})'
            )

        plt.plot([0, 1], [0, 1], 'r--', label='Random Guess (AUC = 0.500)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve Comparison for {stock_ticker}')
        plt.legend(loc='lower right')
        plt.grid(True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f'{stock_ticker}_ROC_Curve_Comparison.png'
            )
        )
        plt.close()
    except Exception as exc:
        print(f'Could not generate ROC curve: {exc}')


def get_up_probability(model, X):
    probabilities = model.predict_proba(X)
    positions = np.where(model.classes_ == 1)[0]

    if len(positions) != 1:
        raise ValueError('Model does not expose class label 1.')

    return probabilities[:, positions[0]]


def evaluate_stock(stock_ticker, all_metrics, all_predictions):
    print(f'\n--- Evaluating {stock_ticker} ---')

    scaler_file = os.path.join(INPUT_DIR, f'{stock_ticker}_scaler.joblib')
    X_test_file = os.path.join(INPUT_DIR, f'{stock_ticker}_X_test.csv')
    y_test_file = os.path.join(INPUT_DIR, f'{stock_ticker}_y_test.csv')
    ensemble_weights_file = os.path.join(
        INPUT_DIR, f'{stock_ticker}_ensemble_weights.joblib'
    )

    try:
        scaler = load(scaler_file)
        ensemble_weights = load(ensemble_weights_file)
        X_test = pd.read_csv(X_test_file, index_col='Date', parse_dates=True)
        y_test = pd.read_csv(
            y_test_file, index_col='Date', parse_dates=True
        ).squeeze("columns")
    except FileNotFoundError as exc:
        print(
            f'Missing required file: {exc.filename}. '
            'Run the updated train.py first.'
        )
        return

    X_test = X_test[FEATURE_COLUMNS]
    X_test_scaled = scaler.transform(X_test)
    y_test = y_test.astype(int)

    baseline_accuracy = y_test.value_counts(normalize=True).max()

    model_probabilities = {}
    individual_reports = {}
    roc_plot_data = {}
    model_accuracies = {}

    # Evaluate each base model.
    for model_name in MODEL_NAMES:
        model_file = os.path.join(
            INPUT_DIR, f'{stock_ticker}_model_{model_name}.joblib'
        )

        if not os.path.exists(model_file):
            print(f'Model file missing; skipping {model_name}.')
            continue

        model = load(model_file)
        y_pred = model.predict(X_test_scaled)
        y_proba = get_up_probability(model, X_test_scaled)
        model_probabilities[model_name] = y_proba

        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[model_name.upper()] = accuracy

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        report_text = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=['Down (0)', 'Up (1)'],
            zero_division=0
        )

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        roc_plot_data[model_name.upper()] = (fpr, tpr, roc_auc)

        individual_reports[model_name] = (
            f'\n==================== {model_name.upper()} '
            f'====================\n'
            f'Accuracy: {accuracy * 100:.2f}%\n'
            f'AUC Score: {roc_auc:.4f}\n\n'
            f'Classification Report:\n{report_text}\n'
            f'Confusion Matrix:\n{cm}\n'
        )

    if not model_probabilities:
        print('No trained models were found.')
        return

    # Use only the learned weights corresponding to available models.
    available_names = list(model_probabilities.keys())
    weights = np.array(
        [ensemble_weights.get(name, 0.0) for name in available_names],
        dtype=float
    )

    if weights.sum() <= 0:
        weights = np.ones(len(available_names), dtype=float)
    weights = weights / weights.sum()

    probability_matrix = np.column_stack(
        [model_probabilities[name] for name in available_names]
    )
    weighted_probability_up = probability_matrix @ weights
    y_pred_ensemble = (weighted_probability_up >= 0.5).astype(int)

    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    model_accuracies[ENSEMBLE_NAME] = ensemble_accuracy

    cm_ensemble = confusion_matrix(
        y_test, y_pred_ensemble, labels=[0, 1]
    )
    report_dict = classification_report(
        y_test,
        y_pred_ensemble,
        labels=[0, 1],
        target_names=['Down (0)', 'Up (1)'],
        zero_division=0,
        output_dict=True
    )
    report_text = classification_report(
        y_test,
        y_pred_ensemble,
        labels=[0, 1],
        target_names=['Down (0)', 'Up (1)'],
        zero_division=0
    )

    fpr, tpr, _ = roc_curve(y_test, weighted_probability_up)
    ensemble_auc = auc(fpr, tpr)
    roc_plot_data[ENSEMBLE_NAME] = (fpr, tpr, ensemble_auc)

    all_metrics.append({
        'Stock': stock_ticker,
        'Model': ENSEMBLE_NAME,
        'Accuracy': ensemble_accuracy,
        'AUC_Score': ensemble_auc,
        'Precision_Up': report_dict['Up (1)']['precision'],
        'Recall_Up': report_dict['Up (1)']['recall'],
        'F1_Score': report_dict['macro avg']['f1-score'],
        'Baseline_Accuracy': baseline_accuracy
    })

    prediction_frame = pd.DataFrame({
        'Date': y_test.index,
        'Stock': stock_ticker,
        'Actual_Move': y_test.values,
        'Predicted_Move': y_pred_ensemble,
        'P_Up': weighted_probability_up,
        'Ensemble_Confidence': np.maximum(
            weighted_probability_up, 1.0 - weighted_probability_up
        ),
        'Correct_Prediction': (
            y_test.values == y_pred_ensemble
        )
    })
    all_predictions.append(prediction_frame)

    save_confusion_matrix(cm_ensemble, stock_ticker, ENSEMBLE_NAME)
    save_roc_curve_plot(roc_plot_data, stock_ticker)

    # Include learned temporal and ensemble weights in the report.
    temporal_file = os.path.join(
        INPUT_DIR, f'{stock_ticker}_temporal_weights.joblib'
    )
    temporal_weights = load(temporal_file) if os.path.exists(temporal_file) else None

    report = [
        f'Weighted ensemble evaluation for {stock_ticker}',
        f'Accuracy: {ensemble_accuracy * 100:.2f}%',
        f'AUC: {ensemble_auc:.4f}',
        f'Baseline accuracy: {baseline_accuracy * 100:.2f}%',
        '',
        'Learned soft-voting weights:'
    ]
    for name in available_names:
        report.append(f'  {name}: {ensemble_weights.get(name, 0.0):.4f}')

    if temporal_weights is not None:
        report.extend([
            '',
            'Learned temporal weights (lag_0 = current window; lag_6 = six sessions ago):',
            '  Tweet: ' + str(np.round(temporal_weights['tweet'], 4).tolist()),
            '  News : ' + str(np.round(temporal_weights['news'], 4).tolist())
        ])

    report.extend([
        '',
        'Ensemble classification report:',
        report_text,
        'Ensemble confusion matrix:',
        str(cm_ensemble),
        '',
        'Individual model reports:'
    ])
    for name in MODEL_NAMES:
        if name in individual_reports:
            report.append(individual_reports[name])

    report_file = os.path.join(
        OUTPUT_DIR, f'{stock_ticker}_evaluation_report_WEIGHTED.txt'
    )
    with open(report_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(report))

    print(f'{stock_ticker} weighted ensemble accuracy: {ensemble_accuracy:.4f}')
    print('Learned soft-voting weights:', ensemble_weights)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_metrics = []
    all_predictions = []

    for stock in STOCKS:
        evaluate_stock(stock, all_metrics, all_predictions)

    if all_metrics:
        pd.DataFrame(all_metrics).to_excel(
            os.path.join(OUTPUT_DIR, 'Weighted_Ensemble_Metrics.xlsx'),
            index=False
        )

    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_excel(
            os.path.join(OUTPUT_DIR, 'Weighted_Ensemble_Predictions.xlsx'),
            index=False
        )

    print('\nWeighted ensemble evaluation complete.')


if __name__ == '__main__':
    main()
