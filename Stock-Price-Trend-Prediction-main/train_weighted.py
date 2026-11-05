import os
import numpy as np
import pandas as pd

from joblib import dump
from scipy.optimize import minimize
from scipy.special import expit, softmax

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
STOCKS = ['AAPL', 'AMZN', 'MSFT', 'NFLX']
TEST_SET_SIZE = 0.20
RANDOM_STATE = 42
LOOKBACK_DAYS = 7

INPUT_DIR = 'Processed_Data'
OUTPUT_DIR = 'Models'

MODEL_NAMES = ['rf', 'svm', 'lr', 'knn', 'mlp']
UNIFORM_WEIGHT = 1.0 / len(MODEL_NAMES)


def get_models_to_train():
    """Return the five base classifiers used by the project."""
    return {
        'rf': RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
        'svm': SVC(
            gamma='auto',
            random_state=RANDOM_STATE,
            class_weight='balanced',
            probability=True
        ),
        'lr': LogisticRegression(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            max_iter=1000
        ),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'mlp': MLPClassifier(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            random_state=RANDOM_STATE,
            max_iter=1000,
            solver='adam'
        )
    }


def make_weekly_lag_features(df):
    """
    Create 7 trading-session observations for each sentiment source.
    lag_0 = sentiment for the current prediction window.
    lag_1 = previous available trading session, ..., lag_6 = six sessions ago.

    This uses the already-computed daily sentiment in Processed_Data.
    FinBERT is NOT rerun by this script.
    """
    result = pd.DataFrame(index=df.index)

    for source in ['Tweet_Sentiment', 'News_Sentiment']:
        for lag in range(LOOKBACK_DAYS):
            result[f'{source}_lag_{lag}'] = df[source].shift(lag)

    return result


def learn_temporal_weights(X_lags, y):
    """
    Learn two non-negative, sum-to-one weight vectors:
      - seven weights for tweet sentiment lags
      - seven weights for news sentiment lags

    The weights are learned by minimizing regularized binary log-loss
    on the early chronological calibration segment only.

    A softmax parameterization guarantees each source's weights sum to 1.
    """
    tweet_cols = [f'Tweet_Sentiment_lag_{i}' for i in range(LOOKBACK_DAYS)]
    news_cols = [f'News_Sentiment_lag_{i}' for i in range(LOOKBACK_DAYS)]

    X_tweet = X_lags[tweet_cols].to_numpy(dtype=float)
    X_news = X_lags[news_cols].to_numpy(dtype=float)
    y_array = np.asarray(y, dtype=float)

    uniform = np.full(LOOKBACK_DAYS, 1.0 / LOOKBACK_DAYS)

    # A one-class calibration segment cannot identify a useful classifier.
    if len(np.unique(y_array)) < 2 or len(y_array) < 10:
        return {'tweet': uniform, 'news': uniform, 'method': 'uniform_fallback'}

    # Parameters: 7 tweet logits + 7 news logits + 2 coefficients + intercept.
    initial = np.zeros(17, dtype=float)

    def objective(params):
        tweet_w = softmax(params[:LOOKBACK_DAYS])
        news_w = softmax(params[LOOKBACK_DAYS:2 * LOOKBACK_DAYS])

        beta_tweet, beta_news, intercept = params[14:17]

        tweet_agg = X_tweet @ tweet_w
        news_agg = X_news @ news_w

        linear = intercept + beta_tweet * tweet_agg + beta_news * news_agg

        # Numerically stable binary cross-entropy.
        data_loss = np.mean(np.logaddexp(0.0, linear) - y_array * linear)

        # Regularize classifier coefficients and gently discourage
        # extreme lag weights when the calibration sample is small.
        coefficient_penalty = 0.01 * (
            beta_tweet ** 2 + beta_news ** 2
        )
        weight_penalty = 0.02 * (
            np.sum((tweet_w - uniform) ** 2) +
            np.sum((news_w - uniform) ** 2)
        )

        return data_loss + coefficient_penalty + weight_penalty

    result = minimize(
        objective,
        initial,
        method='L-BFGS-B',
        options={'maxiter': 1500}
    )

    if not result.success and not np.isfinite(result.fun):
        return {'tweet': uniform, 'news': uniform, 'method': 'uniform_fallback'}

    tweet_w = softmax(result.x[:LOOKBACK_DAYS])
    news_w = softmax(result.x[LOOKBACK_DAYS:2 * LOOKBACK_DAYS])

    return {
        'tweet': tweet_w,
        'news': news_w,
        'method': 'regularized_log_loss_optimization'
    }


def apply_temporal_weights(X_lags, temporal_weights):
    """Convert the 14 lag columns into two learned weighted-average features."""
    tweet_cols = [f'Tweet_Sentiment_lag_{i}' for i in range(LOOKBACK_DAYS)]
    news_cols = [f'News_Sentiment_lag_{i}' for i in range(LOOKBACK_DAYS)]

    tweet_w = np.asarray(temporal_weights['tweet'], dtype=float)
    news_w = np.asarray(temporal_weights['news'], dtype=float)

    output = pd.DataFrame(index=X_lags.index)
    output['Tweet_Sentiment'] = (
        X_lags[tweet_cols].to_numpy(dtype=float) @ tweet_w
    )
    output['News_Sentiment'] = (
        X_lags[news_cols].to_numpy(dtype=float) @ news_w
    )

    return output


def get_up_probability(model, X):
    """Return P(class=1), explicitly locating class label 1."""
    probabilities = model.predict_proba(X)
    class_positions = np.where(model.classes_ == 1)[0]

    if len(class_positions) != 1:
        raise ValueError('Model does not expose class label 1.')

    return probabilities[:, class_positions[0]]


def learn_ensemble_weights(validation_probabilities, y_validation, names):
    """
    Learn non-negative, sum-to-one model weights from a later chronological
    validation segment. A small shrinkage penalty discourages extreme weights.
    """
    matrix = np.column_stack(
        [validation_probabilities[name] for name in names]
    )
    y_array = np.asarray(y_validation, dtype=int)
    uniform = np.full(len(names), 1.0 / len(names))

    if len(np.unique(y_array)) < 2 or len(y_array) < 10:
        return {name: float(uniform[i]) for i, name in enumerate(names)}

    def objective(raw_weights):
        weights = softmax(raw_weights)
        combined = np.clip(matrix @ weights, 1e-7, 1.0 - 1e-7)
        loss = log_loss(y_array, combined, labels=[0, 1])
        shrinkage = 0.05 * np.sum((weights - uniform) ** 2)
        return loss + shrinkage

    result = minimize(
        objective,
        np.zeros(len(names)),
        method='L-BFGS-B',
        options={'maxiter': 1000}
    )

    if not np.isfinite(result.fun):
        return {name: float(uniform[i]) for i, name in enumerate(names)}

    learned = softmax(result.x)
    return {name: float(learned[i]) for i, name in enumerate(names)}


def train_stock(stock_ticker):
    print(f'\n--- Starting training for {stock_ticker} ---')

    input_file = os.path.join(
        INPUT_DIR, f'{stock_ticker}_model_ready.csv'
    )

    try:
        df = pd.read_csv(input_file, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f'Could not find {input_file}. Run preprocessing first.')
        return

    if df.empty:
        print(f'{input_file} is empty. Skipping.')
        return

    required = ['Tweet_Sentiment', 'News_Sentiment', 'Target_Move']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f'Missing required columns for {stock_ticker}: {missing}')
        return

    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')].copy()

    # Create the trailing 7-trading-session feature window from existing
    # daily sentiment values. No raw text or FinBERT inference is needed.
    X_lags = make_weekly_lag_features(df)
    data = pd.concat([df[['Target_Move']], X_lags], axis=1).dropna()

    if len(data) < 40:
        print(
            f'Only {len(data)} rows remain after creating 7-day lags. '
            'At least 40 are recommended; skipping to avoid unstable weights.'
        )
        return

    y_all = data['Target_Move'].astype(int)
    X_lags_all = data.drop(columns=['Target_Move'])

    # Final untouched chronological test segment.
    split_at = int(len(data) * (1.0 - TEST_SET_SIZE))
    X_lags_train = X_lags_all.iloc[:split_at]
    y_train = y_all.iloc[:split_at]
    X_lags_test = X_lags_all.iloc[split_at:]
    y_test = y_all.iloc[split_at:]

    if len(y_train) < 20 or len(y_test) == 0:
        print('Insufficient training/test rows. Skipping.')
        return

    # Inner chronological calibration:
    # first half learns temporal lag weights and fits temporary models;
    # second half learns the soft-voting weights.
    calibration_split = int(len(X_lags_train) * 0.50)
    X_calibration_fit_lags = X_lags_train.iloc[:calibration_split]
    y_calibration_fit = y_train.iloc[:calibration_split]
    X_ensemble_validation_lags = X_lags_train.iloc[calibration_split:]
    y_ensemble_validation = y_train.iloc[calibration_split:]

    temporal_weights = learn_temporal_weights(
        X_calibration_fit_lags,
        y_calibration_fit
    )

    X_weighted_all = apply_temporal_weights(
        X_lags_all,
        temporal_weights
    )
    X_train = X_weighted_all.iloc[:split_at]
    X_test = X_weighted_all.iloc[split_at:]

    X_calibration_fit = X_train.iloc[:calibration_split]
    X_ensemble_validation = X_train.iloc[calibration_split:]

    # Fit temporary models on the earlier calibration segment.
    calibration_scaler = StandardScaler()
    X_calibration_fit_scaled = calibration_scaler.fit_transform(
        X_calibration_fit
    )
    X_validation_scaled = calibration_scaler.transform(
        X_ensemble_validation
    )

    validation_probabilities = {}
    for name, model in get_models_to_train().items():
        try:
            model.fit(X_calibration_fit_scaled, y_calibration_fit)
            validation_probabilities[name] = get_up_probability(
                model, X_validation_scaled
            )
            print(f'  Calibration predictions generated: {name}')
        except Exception as exc:
            print(f'  Could not calibrate {name}: {exc}')

    if not validation_probabilities:
        print('No models could be calibrated. Skipping this stock.')
        return

    available_names = list(validation_probabilities.keys())
    ensemble_weights = learn_ensemble_weights(
        validation_probabilities,
        y_ensemble_validation,
        available_names
    )

    # Refit the final base models on the complete outer training period.
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train)
    X_test_scaled = final_scaler.transform(X_test)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dump(
        final_scaler,
        os.path.join(OUTPUT_DIR, f'{stock_ticker}_scaler.joblib')
    )

    final_models = get_models_to_train()
    successfully_trained = []

    for name, model in final_models.items():
        try:
            model.fit(X_train_scaled, y_train)
            dump(
                model,
                os.path.join(
                    OUTPUT_DIR, f'{stock_ticker}_model_{name}.joblib'
                )
            )
            successfully_trained.append(name)
            print(f'  Saved final model: {name}')
        except Exception as exc:
            print(f'  Final training failed for {name}: {exc}')

    # Retain and renormalize only weights for models that were saved.
    ensemble_weights = {
        name: ensemble_weights.get(name, 0.0)
        for name in successfully_trained
    }
    total_weight = sum(ensemble_weights.values())

    if total_weight <= 0:
        ensemble_weights = {
            name: 1.0 / len(successfully_trained)
            for name in successfully_trained
        }
    else:
        ensemble_weights = {
            name: weight / total_weight
            for name, weight in ensemble_weights.items()
        }

    dump(
        ensemble_weights,
        os.path.join(
            OUTPUT_DIR, f'{stock_ticker}_ensemble_weights.joblib'
        )
    )
    dump(
        temporal_weights,
        os.path.join(
            OUTPUT_DIR, f'{stock_ticker}_temporal_weights.joblib'
        )
    )

    # Save the transformed test features. evaluate.py consumes these directly.
    X_test.to_csv(
        os.path.join(OUTPUT_DIR, f'{stock_ticker}_X_test.csv'),
        index_label='Date'
    )
    y_test.to_csv(
        os.path.join(OUTPUT_DIR, f'{stock_ticker}_y_test.csv'),
        index_label='Date',
        header=['Target_Move']
    )

    # Save a human-readable record of the learned weights.
    weight_rows = []
    for source in ['tweet', 'news']:
        for lag, weight in enumerate(temporal_weights[source]):
            weight_rows.append({
                'Weight_Type': f'{source}_temporal',
                'Component': f'lag_{lag}',
                'Weight': float(weight)
            })

    for name, weight in ensemble_weights.items():
        weight_rows.append({
            'Weight_Type': 'soft_voting',
            'Component': name,
            'Weight': float(weight)
        })

    pd.DataFrame(weight_rows).to_csv(
        os.path.join(OUTPUT_DIR, f'{stock_ticker}_learned_weights.csv'),
        index=False
    )

    print(f'\n{stock_ticker} learned temporal weights:')
    print('  Tweet:', np.round(temporal_weights['tweet'], 4))
    print('  News :', np.round(temporal_weights['news'], 4))
    print('  Soft voting:', {
        key: round(value, 4)
        for key, value in ensemble_weights.items()
    })
    print(
        f'{stock_ticker}: saved {len(successfully_trained)} models, '
        'learned weights, and untouched test data.'
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for stock in STOCKS:
        train_stock(stock)
    print('\nTraining and calibration pipeline complete.')


if __name__ == '__main__':
    main()
