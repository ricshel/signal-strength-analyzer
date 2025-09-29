import os
import math
import yaml
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def fetch_prices(ticker: str, days: int) -> pd.DataFrame:
    # Yahoo allows period like "2y", but using start/end via days keeps it simple
    df = yf.download(ticker, period=f"{days}d", interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    return df.rename(columns=str.lower)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["sma20"]  = df["close"].rolling(20).mean()
    df["sma50"]  = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    return df

def score_row(close, sma20, sma50, sma200, sma50_prev5) -> int:
    """close/sma* are plain floats (or NaN), not Series."""
    score = 0

    # 1) Price vs SMA20
    if pd.notna(sma20):
        score += 2 if close > sma20 else -2

    # 2) SMA20 vs SMA50
    if pd.notna(sma20) and pd.notna(sma50):
        score += 2 if sma20 > sma50 else -2

    # 3) SMA50 vs SMA200
    if pd.notna(sma50) and pd.notna(sma200):
        score += 4 if sma50 > sma200 else -4

    # 4) Slope of SMA50 over last 5 days
    if pd.notna(sma50) and pd.notna(sma50_prev5):
        score += 4 if sma50 > sma50_prev5 else -4

    # 5) Distance from SMA200
    if pd.notna(sma200):
        dist = (close - sma200) / sma200
        if dist >= 0.10:
            score += 8
        elif dist <= -0.10:
            score -= 8

    return int(max(-20, min(20, score)))

def score_ticker(ticker: str, days: int) -> dict:
    df = fetch_prices(ticker, days)
    df = add_indicators(df)

    # need at least 200 + 5 days of history
    if len(df) < 205:
        raise ValueError(f"Not enough history for {ticker} ({len(df)} rows)")

    # pull scalars (not Series)
    close         = float(df["close"].iloc[-1])
    sma20         = float(df["sma20"].iloc[-1])   if pd.notna(df["sma20"].iloc[-1])   else float("nan")
    sma50         = float(df["sma50"].iloc[-1])   if pd.notna(df["sma50"].iloc[-1])   else float("nan")
    sma200        = float(df["sma200"].iloc[-1])  if pd.notna(df["sma200"].iloc[-1])  else float("nan")
    sma50_prev5   = float(df["sma50"].iloc[-6])   if pd.notna(df["sma50"].iloc[-6])   else float("nan")

    s = score_row(close, sma20, sma50, sma200, sma50_prev5)

    return {
        "ticker": ticker,
        "date": df.index[-1].date().isoformat(),
        "close": round(close, 4),
        "sma20":  None if math.isnan(sma20)  else round(sma20, 4),
        "sma50":  None if math.isnan(sma50)  else round(sma50, 4),
        "sma200": None if math.isnan(sma200) else round(sma200, 4),
        "score": s,
    }

def plot_barometer(df_scores: pd.DataFrame, out_path: str):
    # Sort by score descending so strongest at bottom/top (choose your taste)
    df_scores = df_scores.sort_values("score")
    labels = df_scores["ticker"].tolist()
    scores = df_scores["score"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    y = range(len(labels))
    # Plot negative to left, positive to right around zero:
    ax.barh(y, scores)
    ax.set_yticks(y, labels)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Trend Score (−20 … +20)")
    ax.set_title("Trend Barometer")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def write_csv(df_scores: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_scores.to_csv(out_path, index=False)

def ensure_index_html(docs_dir: str, img_name: str, csv_name: str):
    path = os.path.join(docs_dir, "index.html")
    if os.path.exists(path):
        return
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Trend Barometer</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial; margin:24px;}}</style>
</head><body>
<h1>Trend Barometer</h1>
<p>Auto-generated from simple SMA rules (max ±20).</p>
<img src="{img_name}" alt="Trend Barometer" style="max-width:100%;height:auto;">
<p><a href="{csv_name}">Download scores (CSV)</a></p>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

def main():
    cfg = load_config()
    tickers = cfg["tickers"]
    days = int(cfg.get("history_days", 400))
    out_dir = cfg.get("output_dir", "docs")
    img_name = cfg.get("chart_filename", "trend_barometer.png")
    csv_name = cfg.get("csv_filename", "trend_scores.csv")

    results = []
    for t in tickers:
        try:
            results.append(score_ticker(t, days))
        except Exception as e:
            print(f"[WARN] {t}: {e}")

    if not results:
        raise SystemExit("No scores produced.")

    df_scores = pd.DataFrame(results)
    write_csv(df_scores, os.path.join(out_dir, csv_name))
    plot_barometer(df_scores, os.path.join(out_dir, img_name))
    ensure_index_html(out_dir, img_name, csv_name)
    print("Done. Files in:", out_dir)

if __name__ == "__main__":
    main()