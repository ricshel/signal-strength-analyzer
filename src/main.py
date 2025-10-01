#!/usr/bin/env python3
# dashboard_and_barometer.py
# Builds:
#  - Per-ticker PNG charts (Close + SMA20/50/200)
#  - latest_prices_sma.csv
#  - Trend Barometer (±20) PNG from your SMA rules
#  - trend_scores.csv
#  - docs/index.html gallery that includes both sections

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


def plot_barometer(df_scores: pd.DataFrame, out_path: str) -> None:
    # sort so strongest at top
    df_scores = df_scores.sort_values("score")
    labels = df_scores["ticker"].tolist()
    scores = df_scores["score"].tolist()
    y = range(len(labels))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(y, scores)
    ax.set_yticks(y, labels)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Trend Score (−20 … +20)")
    ax.set_title("Trend Barometer")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------- HTML ----------------

def write_gallery(rows_latest: List[dict], docs_dir: str, charts_subdir: str,
                  title: str, baro_img: str, latest_csv: str, scores_csv: str):
    os.makedirs(docs_dir, exist_ok=True)
    rows_sorted = sorted(rows_latest, key=lambda r: r["ticker"].lower())

    cards = []
    for r in rows_sorted:
        img = f"{charts_subdir}/{r['ticker']}.png"
        def fmt(x): return "—" if x is None else x
        cards.append(f"""
        <div class="card">
          <div class="head">
            <h3>{r['ticker']}</h3>
            <div class="date">{r['date']}</div>
          </div>
          <img src="{img}" alt="{r['ticker']} chart" loading="lazy">
          <div class="gridmeta">
            <div>Close</div><b>{fmt(r['close'])}</b>
            <div>SMA20</div><b>{fmt(r['sma20'])}</b>
            <div>SMA50</div><b>{fmt(r['sma50'])}</b>
            <div>SMA200</div><b>{fmt(r['sma200'])}</b>
          </div>
        </div>
        """)

    updated_on = rows_sorted[0]['date'] if rows_sorted else ""

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ --bg:#fff; --fg:#111; --muted:#666; --card:#f9fafb; --border:#e5e7eb; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#0b0d10; --fg:#e7eaee; --muted:#a1a7b0; --card:#12161c; --border:#22272f; }}
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--fg); font: 15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,Arial; margin:16px; }}
h1 {{ margin:8px 0 16px; font-size:22px; }}
section {{ margin-bottom:22px; }}
.gallery {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(360px,1fr)); gap:14px; align-items:start; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:12px; }}
.card img {{ width:100%; height:auto; border-radius:8px; display:block; }}
.head {{ display:flex; align-items:baseline; justify-content:space-between; margin-bottom:8px; }}
h3 {{ margin:0; font-size:16px; font-weight:600; }}
.date {{ color:var(--muted); font-size:12px; }}
.gridmeta {{ display:grid; grid-template-columns:auto 1fr; gap:6px 12px; margin-top:8px; }}
footer {{ margin-top:20px; color:var(--muted); font-size:13px; }}
a {{ color:inherit; }}
img.baro {{ width:100%; max-width:980px; height:auto; display:block; border-radius:10px; }}
.links a {{ margin-right:12px; }}
</style>
</head>
<body>
<h1>{title}</h1>

<section>
  <h2>Trend Barometer (−20 … +20)</h2>
  <img class="baro" src="{baro_img}" alt="Trend Barometer">
  <div class="links">
    <a href="{scores_csv}" download>Download trend scores (CSV)</a>
  </div>
</section>

<section>
  <h2>Per-Ticker Price & SMAs</h2>
  <div class="gallery">
  {''.join(cards)}
  </div>
  <div class="links" style="margin-top:10px;">
    <a href="{latest_csv}" download>Download latest values (CSV)</a>
  </div>
</section>

<footer>
  <div>Auto-generated from Yahoo Finance (adjusted close). Updated on {updated_on}.</div>
</footer>
</body>
</html>"""
    with open(os.path.join(docs_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


# ---------------- Orchestration ----------------

def main():
    cfg = load_config()
    tickers: List[str] = cfg.get("tickers", [])
    if not tickers:
        raise SystemExit("No tickers found in config.yaml (expected key: tickers: [ ... ])")

    days: int = int(cfg.get("history_days", 800))
    docs_dir: str = cfg.get("output_dir", "docs")
    charts_dir: str = os.path.join(docs_dir, "charts")
    lookback: int = int(cfg.get("chart_lookback", 400))
    latest_csv: str = cfg.get("latest_csv", "latest_prices_sma.csv")
    scores_csv: str = cfg.get("scores_csv", "trend_scores.csv")
    baro_img: str = cfg.get("barometer_image", "trend_barometer.png")
    title: str = cfg.get("dashboard_title", "Price & SMA Dashboard")

    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    latest_rows: List[dict] = []
    score_rows: List[dict] = []

    for t in tickers:
        try:
            df = fetch_prices(t, days)
            df = add_indicators(df)

            # Per-ticker chart + latest values
            if len(df) >= 20:
                plot_ticker(df, t, charts_dir, lookback=lookback)
                latest_rows.append(latest_row(df, t))
            else:
                print(f"[WARN] {t}: not enough history to plot")

            # Scoring (needs 200 + 5)
            try:
                score_rows.append(score_from_df(t, df))
            except Exception as e:
                print(f"[WARN] {t}: scoring skipped — {e}")

        except Exception as e:
            print(f"[WARN] {t}: {e}")

    if not latest_rows and not score_rows:
        raise SystemExit("No outputs produced.")

    # Write CSVs
    if latest_rows:
        pd.DataFrame(latest_rows).sort_values("ticker").to_csv(
            os.path.join(docs_dir, latest_csv), index=False
        )
    if score_rows:
        df_scores = pd.DataFrame(score_rows).sort_values("ticker")
        df_scores.to_csv(os.path.join(docs_dir, scores_csv), index=False)
        # Barometer
        plot_barometer(df_scores[["ticker", "score"]].copy(), os.path.join(docs_dir, baro_img))

    # HTML
    write_gallery(
        rows_latest=latest_rows,
        docs_dir=docs_dir,
        charts_subdir=os.path.basename(charts_dir),
        title=title,
        baro_img=baro_img,
        latest_csv=latest_csv,
        scores_csv=scores_csv,
    )

    print(f"Done. Open {os.path.join(docs_dir, 'index.html')}")


if __name__ == "__main__":
    main()