import pandas as pd
import numpy as np
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

# --- Configuration ---
STOCKS = ['AAPL', 'AMZN', 'MSFT', 'NFLX']
TRADING_OPEN_HOUR = 9
TRADING_OPEN_MINUTE = 30
BATCH_SIZE = 32

# --- File Paths ---
INPUT_TWEET_DIR = 'Twitter'
INPUT_NEWS_DIR = 'News'
INPUT_STOCK_DIR = 'Stock'
OUTPUT_DIR = 'Processed_Data'


# --- 1. Load the FinBERT Model ---

def load_transformer_model():
    """Loads the FinBERT model and tokenizer."""
    print("Loading FinBERT model... (This may take a moment)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (If this is CPU, preprocessing will be very slow)")
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    print("FinBERT model loaded successfully.")
    return model, tokenizer, device


# --- 2. New Sentiment Analysis Functions ---

def get_finbert_sentiment(texts, model, tokenizer, device):
    """Processes a list of texts in batches and returns (pos - neg) sentiment scores."""
    all_scores = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        inputs = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = softmax(outputs.logits, dim=1)
        positive_probs = probs[:, 0]
        negative_probs = probs[:, 1]
        scores = positive_probs - negative_probs
        all_scores.extend(scores.cpu().numpy())

        if (i // BATCH_SIZE) % 50 == 0:
            print(f"  Processed batch {i // BATCH_SIZE} / {len(texts) // BATCH_SIZE}")

    return all_scores


def preprocess_text_data(filepath, text_col, date_col, time_col, model, tokenizer, device):
    """Loads text data, creates timestamp, and calculates FinBERT sentiment."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found {filepath}. Skipping.")
        return None
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    required_cols = [text_col, date_col, time_col]
    if not all(col in df.columns for col in required_cols):
        print(f"File {filepath} missing one of {required_cols}. Skipping.")
        return None

    df.dropna(subset=[text_col, date_col, time_col], inplace=True)

    try:
        df['timestamp'] = pd.to_datetime(df[date_col] + ' ' + df[time_col],
                                         format='%Y/%m/%d %H:%M:%S',
                                         errors='coerce')
    except Exception:
        df['timestamp'] = pd.to_datetime(df[date_col] + ' ' + df[time_col],
                                         errors='coerce')

    df.dropna(subset=['timestamp'], inplace=True)
    if df.empty:
        print(f"No valid data in {filepath} after timestamp cleaning.")
        return None

    df.set_index('timestamp', inplace=True)

    print(f"Starting FinBERT sentiment analysis for {filepath}...")
    print(f"This will take a long time. {len(df)} texts to process.")

    texts_to_process = df[text_col].astype(str).tolist()
    sentiment_scores = get_finbert_sentiment(texts_to_process, model, tokenizer, device)
    df['sentiment'] = sentiment_scores

    print(f"Successfully processed {filepath} with FinBERT.")
    return df[['sentiment']]


# --- 3. Aggregation Function (with Holiday Fix) ---

def aggregate_sentiment(df, open_hour, open_minute):
    """
    Aggregates sentiment scores based on the "9:30 AM to 9:30 AM" window.
    Uses 'B' (Business Day) to roll forward weekend/holiday sentiment.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['Agg_Sentiment'])

    offset = pd.Timedelta(hours=open_hour, minutes=open_minute)
    df_shifted = df.set_index(df.index - offset)

    # 'B' (Business Day) correctly sums and rolls forward
    # sentiment from non-business days (weekends, holidays).
    df_agg = df_shifted.resample('B').sum()

    df_agg = df_agg.rename(columns={'sentiment': 'Agg_Sentiment'})
    df_agg.index.name = 'Date'
    return df_agg


# --- 4. Stock Processing Function (Sentiment-Only) ---

def process_stock(stock_ticker, model, tokenizer, device):
    """Runs the full preprocessing pipeline for a single stock."""
    print(f"\n--- Processing {stock_ticker} ---")

    # 1. Process Tweets (Now uses FinBERT)
    tweet_file = os.path.join(INPUT_TWEET_DIR, f"{stock_ticker}_tweets.csv")
    df_tweets_senti = preprocess_text_data(
        filepath=tweet_file, text_col='tweet', date_col='date', time_col='time',
        model=model, tokenizer=tokenizer, device=device
    )
    df_tweets_agg = aggregate_sentiment(df_tweets_senti, TRADING_OPEN_HOUR, TRADING_OPEN_MINUTE)
    df_tweets_agg = df_tweets_agg.rename(columns={'Agg_Sentiment': 'Tweet_Sentiment'})

    # 2. Process News (Now uses FinBERT)
    news_file = os.path.join(INPUT_NEWS_DIR, f"{stock_ticker}_news.csv")
    df_news_senti = preprocess_text_data(
        filepath=news_file, text_col='Complete_Title', date_col='Time', time_col='Exact_Time',
        model=model, tokenizer=tokenizer, device=device
    )
    df_news_agg = aggregate_sentiment(df_news_senti, TRADING_OPEN_HOUR, TRADING_OPEN_MINUTE)
    df_news_agg = df_news_agg.rename(columns={'Agg_Sentiment': 'News_Sentiment'})

    # 3. Load Stock Data (Sentiment-Only: We only keep the target)
    stock_file = os.path.join(INPUT_STOCK_DIR, f"{stock_ticker}.csv")
    if not os.path.exists(stock_file):
        print(f"Error: Stock file not found {stock_file}. Cannot proceed.")
        return

    try:
        df_stock = pd.read_csv(stock_file)
        df_stock['Date'] = pd.to_datetime(df_stock['Date'], format='%Y/%m/%d')
        df_stock.set_index('Date', inplace=True)

        target_col = 'Goal II: Open t+1 - Open t'

        if target_col not in df_stock.columns:
            print(f"Target column '{target_col}' not in {stock_file}. Skipping.")
            return

        # We only keep the target column
        df_stock_model = df_stock[[target_col]].copy()

        df_stock_model.rename(columns={target_col: 'Target_Move'}, inplace=True)
        df_stock_model.dropna(subset=['Target_Move'], inplace=True)
        df_stock_model['Target_Move'] = df_stock_model['Target_Move'].apply(lambda x: 1 if x == 1 else 0)

    except Exception as e:
        print(f"Failed to load or process stock file {stock_file}: {e}")
        return

    # 4. Merge All Data (Unchanged)
    df_final = df_stock_model
    df_final = df_final.join(df_tweets_agg, how='left')
    df_final = df_final.join(df_news_agg, how='left')

    df_final['Tweet_Sentiment'] = df_final['Tweet_Sentiment'].fillna(0)
    df_final['News_Sentiment'] = df_final['News_Sentiment'].fillna(0)

    # Drop any rows where we might be missing technical data
    df_final.dropna(inplace=True)

    # 5. Save Final File (Unchanged)
    output_file = os.path.join(OUTPUT_DIR, f"{stock_ticker}_model_ready.csv")
    df_final.to_csv(output_file)
    print(f"Successfully created {output_file} with {len(df_final)} rows.")
    print("Final data columns:", df_final.columns)


# --- 5. Main Execution ---

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, tokenizer, device = load_transformer_model()

    for stock in STOCKS:
        process_stock(stock, model, tokenizer, device)

    print("\nPreprocessing complete with FinBERT and Sentiment-Only features.")


if __name__ == "__main__":
    main()
