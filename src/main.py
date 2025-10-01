#!/usr/bin/env python3

from __future__ import annotations
import os
import math
import warnings
from typing import List, Dict

import yaml
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---------------- Config / IO ----------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------- Data + Indicators ----------------

def fetch_prices(ticker: str, days: int) -> pd.DataFrame:
    # yfinance changed auto_adjust default to True; set explicitly to be clear.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        df = yf.download(ticker, period=f"{days}d", interval="1d",
                         auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker} (days={days})")
    df = df.rename(columns=str.lower)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    s = df["close"]
    df["sma20"]  = s.rolling(20,  min_periods=20).mean()
    df["sma50"]  = s.rolling(50,  min_periods=50).mean()
    df["sma200"] = s.rolling(200, min_periods=200).mean()
    return df


def latest_row(df: pd.DataFrame, ticker: str) -> dict:
    last = df.iloc[-1]
    def _rnd(x):
        return None if (isinstance(x, float) and math.isnan(x)) else (round(float(x), 6) if x is not None else None)
    return {
        "ticker": ticker,
        "date": df.index[-1].date().isoformat(),
        "close": _rnd(last.get("close")),
        "sma20": _rnd(last.get("sma20")),
        "sma50": _rnd(last.get("sma50")),
        "sma200": _rnd(last.get("sma200")),
    }


# ---------------- Scoring (±20) ----------------

def score_row(close, sma20, sma50, sma200, sma50_prev5) -> int:
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


def score_from_df(ticker: str, df: pd.DataFrame) -> dict:
    if len(df) < 205:
        raise ValueError(f"Not enough history for {ticker} ({len(df)} rows)")
    close       = float(df["close"].iloc[-1])
    sma20       = float(df["sma20"].iloc[-1])   if pd.notna(df["sma20"].iloc[-1]) else float("nan")
    sma50       = float(df["sma50"].iloc[-1])   if pd.notna(df["sma50"].iloc[-1]) else float("nan")
    sma200      = float(df["sma200"].iloc[-1])  if pd.notna(df["sma200"].iloc[-1]) else float("nan")
    sma50_prev5 = float(df["sma50"].iloc[-6])   if pd.notna(df["sma50"].iloc[-6]) else float("nan")
    s = score_row(close, sma20, sma50, sma200, sma50_prev5)
    return {
        "ticker": ticker,
        "date": df.index[-1].date().isoformat(),
        "close": round(close, 6),
        "sma20":  None if math.isnan(sma20)  else round(sma20, 6),
        "sma50":  None if math.isnan(sma50)  else round(sma50, 6),
        "sma200": None if math.isnan(sma200) else round(sma200, 6),
        "score": s,
    }


# ---------------- Plotting ----------------

def plot_ticker(df: pd.DataFrame, ticker: str, out_dir: str, lookback: int = 400) -> str:
    os.makedirs(out_dir, exist_ok=True)
    dd = df.tail(lookback).copy()
    fig, ax = plt.subplots(figsize=(8.5, 3.3))
    ax.plot(dd.index, dd["close"], label="Close", linewidth=1.5)
    if dd["sma20"].notna().any():  ax.plot(dd.index, dd["sma20"],  label="SMA20",  linewidth=1.0)
    if dd["sma50"].notna().any():  ax.plot(dd.index, dd["sma50"],  label="SMA50",  linewidth=1.0)
    if dd["sma200"].notna().any(): ax.plot(dd.index, dd["sma200"], label="SMA200", linewidth=1.0)
    ax.set_title(f"{ticker} — Price & SMAs (last {min(len(dd), lookback)} sessions)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=4, fontsize=8, frameon=False)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{ticker}.png")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_barometer(rows_scores, out_path: str) -> None:
    # Accept either list[dict] or DataFrame
    if isinstance(rows_scores, list):
        df_scores = pd.DataFrame(rows_scores)
    else:
        df_scores = rows_scores.copy()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # keep only computed scores
    if "score" not in df_scores.columns:
        df_scores["score"] = None
    df_scores = df_scores[df_scores["score"].notna()]

    if df_scores.empty:
        plt.figure(figsize=(8, 2))
        plt.text(0.5, 0.5, "No scores (not enough history)", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        return

    df_scores = df_scores.sort_values("score")
    labels = df_scores["ticker"].tolist()
    scores = df_scores["score"].tolist()
    y = range(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y, scores)
    ax.set_yticks(y, labels)
    ax.axvline(0, linewidth=1)
    ax.set_xlim(-20, 20)
    ax.set_xlabel("Trend Score (−20 … +20)")
    ax.set_title("Trend Barometer")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

# ---------------- HTML ----------------

def write_page_single_image(title: str, docs_dir: str, baro_img: str, updated_on: str):
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ --bg:#fff; --fg:#111; --muted:#666; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#0b0d10; --fg:#e7eaee; --muted:#a1a7b0; }}
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,Arial; margin:16px; }}
h1 {{ margin:8px 0 16px; font-size:22px; }}
img.baro {{ width:100%; max-width:980px; height:auto; display:block; border-radius:10px; }}
footer {{ margin-top:16px; color:var(--muted); font-size:12px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<img class="baro" src="{baro_img}" alt="Trend Barometer">
<footer>Auto-generated from Yahoo Finance (adjusted close). Updated on {updated_on}.</footer>
</body>
</html>"""
    os.makedirs(docs_dir, exist_ok=True)
    with open(os.path.join(docs_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

# ---------------- Orchestration ----------------
def main():
    cfg = load_config()

    # Output folder for branch-deploy Pages
    docs_dir: str = cfg.get("output_dir") or cfg.get("site_dir") or "docs"
    os.makedirs(docs_dir, exist_ok=True)

    tickers: List[str] = cfg.get("tickers", [])
    if not tickers:
        raise SystemExit("No tickers found in config.yaml (expected key: tickers: [ ... ])")

    days: int       = int(cfg.get("history_days", 800))
    baro_img: str   = cfg.get("barometer_image", "trend_barometer.png")
    title: str      = cfg.get("dashboard_title", "Trend Barometer")

    print(f"[INFO] Using config: {cfg.get('_cfg_path')}")
    print(f"[INFO] docs_dir={docs_dir}")
    print(f"[INFO] Tickers ({len(tickers)}): {', '.join(tickers)}")

    # Compute scores only (no per-ticker charts/CSVs)
    score_rows: List[dict] = []
    updated_on = ""
    for t in tickers:
        try:
            df = fetch_prices(t, days)
            df = add_indicators(df)
            # remember a date just to show "Updated on" (use the latest df date)
            if not updated_on:
                updated_on = df.index[-1].date().isoformat()
            score_rows.append(score_from_df(t, df))
        except Exception as e:
            print(f"[WARN] {t}: {e}")

    if not score_rows:
        raise SystemExit("No scores produced.")

    # Save the barometer image into docs/
    baro_path = os.path.join(docs_dir, baro_img)
    plot_barometer(score_rows, baro_path)

    # Minimal HTML with only the barometer
    write_page_single_image(title, docs_dir, baro_img, updated_on or "")

    print(f"[DONE] HTML: {os.path.join(docs_dir, 'index.html')}")
    print(f"[DONE] Image: {baro_path}")

if __name__ == "__main__":
    main()