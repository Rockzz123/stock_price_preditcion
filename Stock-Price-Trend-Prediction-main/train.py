import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
import sys
import os

# Import our 5 chosen models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# --- Configuration ---
STOCKS = ['AAPL', 'AMZN', 'MSFT', 'NFLX']
TEST_SET_SIZE = 0.2
RANDOM_STATE = 42

# --- File Paths ---
INPUT_DIR = 'Processed_Data'
OUTPUT_DIR = 'Models'


def get_models_to_train():
    """Returns a dictionary of our 5 final models."""

    return {
        'rf': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced'),

        'svm': SVC(gamma='auto', random_state=RANDOM_STATE, class_weight='balanced', probability=True),

        'lr': LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced', max_iter=1000),

        'knn': KNeighborsClassifier(n_neighbors=5),

        # Our chosen best MLP architecture
        'mlp': MLPClassifier(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            random_state=RANDOM_STATE,
            max_iter=1000,
            solver='adam'
        )
    }


def train_stock(stock_ticker):
    """Loads data, trains all 5 models, and saves them."""
    print(f"\n--- Starting Training for {stock_ticker} ---")

    input_file = os.path.join(INPUT_DIR, f"{stock_ticker}_model_ready.csv")

    try:
        df = pd.read_csv(input_file, index_col='Date')
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Did you run preprocess.py first?")
        return

    if df.empty:
        print(f"Error: {input_file} is empty. Skipping.")
        return

    # 1. Define X and y
    y = df['Target_Move']
    X = df[['Tweet_Sentiment', 'News_Sentiment']]

    # 2. Train/Test Split (No shuffle for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SET_SIZE,
        random_state=RANDOM_STATE,
        shuffle=False
    )

    if X_train.empty or X_test.empty:
        print("Error: Not enough data for train/test split. Skipping.")
        return

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size:     {len(X_test)}")

    # 3. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    dump(scaler, os.path.join(OUTPUT_DIR, f'{stock_ticker}_scaler.joblib'))

    # 4. Train All Models
    models_to_train = get_models_to_train()

    for model_name, model in models_to_train.items():
        print(f"  Training {model_name}...")
        try:
            model.fit(X_train_scaled, y_train)
            dump(model, os.path.join(OUTPUT_DIR, f'{stock_ticker}_model_{model_name}.joblib'))
            print(f"  Saved {model_name} model.")
        except Exception as e:
            print(f"  Failed to train {model_name}: {e}")

    # 5. Save Test Data
    X_test.to_csv(os.path.join(OUTPUT_DIR, f'{stock_ticker}_X_test.csv'))
    y_test.to_csv(os.path.join(OUTPUT_DIR, f'{stock_ticker}_y_test.csv'))

    print(f"Model training and test data saving complete for {stock_ticker}.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for stock in STOCKS:
        train_stock(stock)
    print("\nAll training complete.")


if __name__ == "__main__":
    main()