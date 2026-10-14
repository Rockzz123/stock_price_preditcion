import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from joblib import load
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
STOCKS = ['AAPL', 'AMZN', 'MSFT', 'NFLX']
MODEL_NAMES = [
    'rf', 'svm', 'lr', 'knn',  # Our 4 classical models
    'mlp'  # Our 1 chosen MLP
]
ENSEMBLE_NAME = 'ENSEMBLE'

# --- CORRECTED: Use only the 2 Sentiment Features ---
FEATURE_COLUMNS = ['Tweet_Sentiment', 'News_Sentiment']

# --- File Paths ---
INPUT_DIR = 'Models'
OUTPUT_DIR = 'Results'


def evaluate_stock(stock_ticker, all_metrics_list, all_predictions_list):
    """
    Loads models, evaluates them, collects metrics, AND collects row-by-row predictions.
    """
    print(f"\n--- Evaluating Models for {stock_ticker} ---")

    scaler_file = os.path.join(INPUT_DIR, f'{stock_ticker}_scaler.joblib')
    X_test_file = os.path.join(INPUT_DIR, f'{stock_ticker}_X_test.csv')
    y_test_file = os.path.join(INPUT_DIR, f'{stock_ticker}_y_test.csv')

    try:
        # Load scaler and test data
        scaler = load(scaler_file)
        X_test = pd.read_csv(X_test_file, index_col='Date')
        y_test = pd.read_csv(y_test_file, index_col='Date').squeeze()
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}. Did you run train.py first?")
        return

    # --- Ensure X_test has the correct columns before scaling ---
    if not all(col in X_test.columns for col in FEATURE_COLUMNS):
        print(f"Error: X_test file for {stock_ticker} is missing required features {FEATURE_COLUMNS}.")
        return
    X_test = X_test[FEATURE_COLUMNS]

    # Scale X_test
    X_test_scaled = scaler.transform(X_test)

    # Baseline accuracy
    baseline_accuracy = y_test.value_counts(normalize=True).max()

    model_accuracies = {}
    all_soft_preds = {}

    individual_reports = {}
    roc_plot_data = {}

    # --- 1. Evaluate Individual Models ---
    print("  Evaluating Individual Models...")
    for model_name in MODEL_NAMES:
        model_file = os.path.join(INPUT_DIR, f'{stock_ticker}_model_{model_name}.joblib')

        try:
            model = load(model_file)
        except FileNotFoundError:
            print(f"  Warning: Model file {model_file} not found. Skipping.")
            continue

        # Get Hard predictions
        y_pred = model.predict(X_test_scaled)

        # Get Soft predictions
        y_proba = np.zeros(len(y_test))
        try:
            y_proba_all_classes = model.predict_proba(X_test_scaled)
            y_proba = y_proba_all_classes[:, 1]  # Probability of "1" (UP)
            all_soft_preds[model_name] = y_proba
        except Exception as e:
            y_proba = y_pred
            all_soft_preds[model_name] = y_pred

            # Metrics & Reporting (Same as before)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[model_name.upper()] = accuracy
        cm = confusion_matrix(y_test, y_pred)
        report_text = classification_report(y_test, y_pred, target_names=['Down (0)', 'Up (1)'], zero_division=0)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        roc_plot_data[model_name.upper()] = (fpr, tpr, roc_auc)

        report_chunk = f"\n==================== {model_name.upper()} ====================\n"
        report_chunk += f"Accuracy: {accuracy * 100:.2f}%\n"
        report_chunk += f"AUC Score: {roc_auc:.4f}\n\n"
        report_chunk += "Classification Report:\n" + report_text
        report_chunk += "\nConfusion Matrix:\n" + str(cm) + "\n"
        individual_reports[model_name] = report_chunk

        save_confusion_matrix(cm, stock_ticker, model_name.upper())

    # --- 2. Evaluate Ensemble Model ---
    print("  Evaluating Ensemble Model...")

    # Soft Voting
    df_soft_preds = pd.DataFrame(all_soft_preds)
    avg_proba = df_soft_preds.mean(axis=1)
    y_pred_soft_vote = (avg_proba > 0.5).astype(int)
    soft_vote_accuracy = accuracy_score(y_test, y_pred_soft_vote)

    model_accuracies[ENSEMBLE_NAME] = soft_vote_accuracy

    # --- NEW: Collect Prediction Data for this Stock ---
    # We create a DataFrame for the predictions
    df_predictions = pd.DataFrame({
        'Date': y_test.index,
        'Stock': stock_ticker,
        'Actual_Move': y_test.values,
        'Predicted_Move': y_pred_soft_vote,
        'Ensemble_Confidence': avg_proba,
        'Correct_Prediction': (y_test.values == y_pred_soft_vote)
    })
    all_predictions_list.append(df_predictions)

    # Metrics & Reports
    cm_ensemble = confusion_matrix(y_test, y_pred_soft_vote)
    report_dict_ens = classification_report(y_test, y_pred_soft_vote, target_names=['Down (0)', 'Up (1)'],
                                            zero_division=0, output_dict=True)
    report_text_ens = classification_report(y_test, y_pred_soft_vote, target_names=['Down (0)', 'Up (1)'],
                                            zero_division=0)

    fpr_ens, tpr_ens, _ = roc_curve(y_test, avg_proba)
    roc_auc_ens = auc(fpr_ens, tpr_ens)
    roc_plot_data[ENSEMBLE_NAME] = (fpr_ens, tpr_ens, roc_auc_ens)

    # Store Ensemble Metrics for Summary Excel
    all_metrics_list.append({
        'Stock': stock_ticker,
        'Model': ENSEMBLE_NAME,
        'Accuracy': soft_vote_accuracy,
        'AUC_Score': roc_auc_ens,
        'Precision_Up': report_dict_ens['Up (1)']['precision'],
        'Recall_Up': report_dict_ens['Up (1)']['recall'],
        'F1_Score': report_dict_ens['macro avg']['f1-score'],
        'Baseline_Accuracy': baseline_accuracy
    })

    # Text Report Generation
    ensemble_report_text = f"--- Main Ensemble Model ---\n"
    ensemble_report_text += f"==================== {ENSEMBLE_NAME} ====================\n"
    ensemble_report_text += f"Accuracy: {soft_vote_accuracy * 100:.2f}%\n"
    ensemble_report_text += f"AUC Score: {roc_auc_ens:.4f}\n\n"
    ensemble_report_text += "Classification Report:\n" + report_text_ens
    ensemble_report_text += "\nConfusion Matrix:\n" + str(cm_ensemble) + "\n"

    save_confusion_matrix(cm_ensemble, stock_ticker, ENSEMBLE_NAME)
    save_roc_curve_plot(roc_plot_data, stock_ticker)

    # --- Assemble Final Text Report ---
    summary_text = "--- Accuracy Summary ---\n"
    sorted_accuracies = sorted(model_accuracies.items(), key=lambda item: item[1], reverse=True)
    for model_name, acc in sorted_accuracies:
        summary_text += f"{model_name}: {acc * 100:.2f}%\n"

    baseline_text = "--- Baseline Accuracy ---\n"
    baseline_text += f"Baseline Accuracy (Guessing): {baseline_accuracy * 100:.2f}%\n"
    baseline_text += "(Model must be > this value to be useful)\n\n"

    final_report_text = f"--- Model Evaluation Comparison for {stock_ticker} ---\n\n"
    final_report_text += ensemble_report_text + "\n"
    final_report_text += baseline_text
    final_report_text += summary_text + "\n"
    final_report_text += "--- Individual Model Details ---\n"
    for model_name in MODEL_NAMES:
        if model_name in individual_reports:
            final_report_text += individual_reports[model_name]

    report_file = os.path.join(OUTPUT_DIR, f'{stock_ticker}_evaluation_report_ALL_MODELS.txt')
    with open(report_file, 'w') as f:
        f.write(final_report_text)

    print(f"  Saved detailed text report to {report_file}")


def save_confusion_matrix(cm, stock_ticker, model_name):
    """Helper function to save a confusion matrix plot."""
    if(model_name!='ENSEMBLE'):
         return
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Down', 'Predicted Up'],
                    yticklabels=['Actual Down', 'Actual Up'])
        plt.title(f'Confusion Matrix for {stock_ticker} ({model_name})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plot_file = os.path.join(OUTPUT_DIR, f'{stock_ticker}_confusion_matrix_{model_name}.png')
        plt.savefig(plot_file)
        plt.close()
    except Exception as e:
        print(f"  Could not generate plot for {model_name}: {e}")


def save_roc_curve_plot(roc_plot_data, stock_ticker):
    """Saves a single plot with all model ROC curves."""
    try:
        plt.figure(figsize=(10, 8))
        sorted_data = sorted(roc_plot_data.items(), key=lambda item: item[1][2], reverse=True)
        for model_name, (fpr, tpr, roc_auc) in sorted_data:
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', label='Random Guess (AUC = 0.500)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve Comparison for {stock_ticker}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plot_file = os.path.join(OUTPUT_DIR, f'{stock_ticker}_ROC_Curve_Comparison.png')
        plt.savefig(plot_file)
        plt.close()
    except Exception as e:
        print(f"  Could not generate ROC plot: {e}")


def main():
    # Create the output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics = []
    all_predictions = []

    for stock in STOCKS:
        evaluate_stock(stock, all_metrics, all_predictions)

    # --- 1. Save Summary Metrics Excel ---
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        metrics_file = os.path.join(OUTPUT_DIR, 'Final_Model_Metrics.xlsx')
        try:
            df_metrics.to_excel(metrics_file, index=False)
            print(f"\nSuccessfully saved Metrics Excel to: {metrics_file}")
        except Exception as e:
            print(f"Error saving Metrics Excel: {e}")
            df_metrics.to_csv(os.path.join(OUTPUT_DIR, 'Final_Model_Metrics.csv'), index=False)

    # --- 2. Save Predictions Excel (NEW) ---
    if all_predictions:
        df_preds = pd.concat(all_predictions, ignore_index=False)  # Keep Date index? No, reset it for excel
        preds_file = os.path.join(OUTPUT_DIR, 'Predicted_Values.xlsx')
        try:
            df_preds.to_excel(preds_file, index=False)
            print(f"Successfully saved Predictions Excel to: {preds_file}")
        except Exception as e:
            print(f"Error saving Predictions Excel: {e}")
            df_preds.to_csv(os.path.join(OUTPUT_DIR, 'Predicted_Values.csv'), index=False)

    print("\nAll evaluation complete. Check the 'Results' folder.")


if __name__ == "__main__":
    main()