#!/usr/bin/env python3
# dashboard.py
# Build a static dashboard: per-ticker chart (Close + SMA20/50/200),
# CSV of latest values, and an index.html gallery (GitHub Pages friendly).

from __future__ import annotations

import os
import math
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any

import yaml
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---------- Config / IO ----------

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------- Data + Indicators ----------

def fetch_prices(ticker: str, days: int) -> pd.DataFrame:
    """
    Fetch daily prices for `days` back. Uses auto_adjust=True so 'Close' is adjusted for splits/dividends
    (for equities/ETFs; FX/indices/yields unaffected).
    """
    # Silence yfinance auto_adjust warning in case user has older code around.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        df = yf.download(ticker, period=f"{days}d", interval="1d",
                         auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data for {ticker} (days={days})")

    # Normalize column names to lower case (Close -> close)
    df = df.rename(columns=str.lower)

    # Ensure DatetimeIndex ascending and drop duplicate dates if any
    df = df[~df.index.duplicated(keep="last")].sort_index()

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    s = df["close"]
    df["sma20"] = s.rolling(20, min_periods=20).mean()
    df["sma50"] = s.rolling(50, min_periods=50).mean()
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


# ---------- Plotting ----------

def plot_ticker(df: pd.DataFrame, ticker: str, out_dir: str, lookback: int = 400) -> str:
    """
    Saves a PNG chart: Close + SMA20/50/200 over the last `lookback` sessions.
    Returns the file path.
    """
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

    # Nice date formatting
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{ticker}.png")
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------- HTML ----------

def write_gallery(rows: List[dict], docs_dir: str, charts_subdir: str = "charts",
                  title: str = "Price & SMA Dashboard"):
    """
    Writes docs/index.html that shows a responsive grid of per-ticker charts
    with latest Close/SMA20/SMA50/SMA200 values.
    """
    os.makedirs(docs_dir, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: r["ticker"].lower())

    # Build cards
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
:root {{
  --bg:#fff; --fg:#111; --muted:#666; --card:#f9fafb; --border:#e5e7eb;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#0b0d10; --fg:#e7eaee; --muted:#a1a7b0; --card:#12161c; --border:#22272f; }}
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--fg); font: 15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,Arial; margin:16px; }}
h1 {{ margin: 8px 0 16px; font-size: 22px; }}
.gallery {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(360px,1fr)); gap:14px; align-items:start; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:12px; }}
.card img {{ width:100%; height:auto; border-radius:8px; display:block; }}
.head {{ display:flex; align-items:baseline; justify-content:space-between; margin-bottom:8px; }}
h3 {{ margin:0; font-size:16px; font-weight:600; }}
.date {{ color:var(--muted); font-size:12px; }}
.gridmeta {{ display:grid; grid-template-columns:auto 1fr; gap:6px 12px; margin-top:8px; }}
footer {{ margin-top:20px; color:var(--muted); font-size:13px; }}
a {{ color:inherit; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="gallery">
{''.join(cards)}
</div>
<footer>
  <div>Auto-generated from Yahoo Finance (adjusted close). Updated on {updated_on}.</div>
  <div><a href="latest_prices_sma.csv" download>Download latest values (CSV)</a></div>
</footer>
</body>
</html>"""

    with open(os.path.join(docs_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


# ---------- Orchestration ----------

def main():
    cfg = load_config()
    tickers: List[str] = cfg.get("tickers", [])
    if not tickers:
        raise SystemExit("No tickers found in config.yaml (expected key: tickers: [ ... ])")

    days: int = int(cfg.get("history_days", 500))  # show enough to cover SMA200 and chart lookback
    docs_dir: str = cfg.get("output_dir", "docs")
    charts_dir: str = os.path.join(docs_dir, "charts")
    lookback: int = int(cfg.get("chart_lookback", 400))  # sessions to display per chart
    csv_name: str = cfg.get("csv_filename", "latest_prices_sma.csv")
    title: str = cfg.get("dashboard_title", "Price & SMA Dashboard")

    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    rows: List[dict] = []

    for t in tickers:
        try:
            df = fetch_prices(t, days)
            df = add_indicators(df)

            if len(df) < 200:
                raise ValueError(f"Not enough history for {t} ({len(df)} rows)")

            # Save chart and collect latest values
            plot_ticker(df, t, charts_dir, lookback=lookback)
            rows.append(latest_row(df, t))

        except Exception as e:
            print(f"[WARN] {t}: {e}")

    if not rows:
        raise SystemExit("No outputs produced.")

    # CSV of latest close + SMAs
    df_out = pd.DataFrame(rows).sort_values("ticker")
    df_out.to_csv(os.path.join(docs_dir, csv_name), index=False)

    # HTML gallery
    write_gallery(rows, docs_dir, charts_subdir="charts", title=title)

    print(f"Done. Open {os.path.join(docs_dir, 'index.html')}")

if __name__ == "__main__":
    main()
