import argparse
from datetime import date

import pandas as pd

from ssvi import chain, clean, config, metrics, plots, signals, ssvi, stocks, store
from ssvi.polygon_client import PolygonClient


def _process_ticker(client, ticker: str, asof_ts: pd.Timestamp,
                    history: pd.DataFrame) -> dict:
    chain_df = chain.fetch_chain(client, ticker)
    if chain_df.empty:
        raise RuntimeError("empty chain")
    prepared = clean.prepare(chain_df, asof=asof_ts)
    if prepared.empty:
        raise RuntimeError("no liquid contracts after filtering")
    surface = ssvi.fit_surface(prepared)
    closes = stocks.fetch_daily_closes(client, ticker)
    rv20 = stocks.realized_vol(closes)
    spot = float(closes.iloc[-1]) if len(closes) else float(prepared["forward"].iloc[0])
    m = metrics.ticker_metrics(surface, rv20=rv20, spot=spot)
    m["underlying"] = ticker
    m["iv_rank"] = store.metric_rank(history, ticker, "iv30", m["iv30"])
    m["iv1y_rank"] = store.metric_rank(history, ticker, "iv1y", m["iv1y"])
    m.update(signals.suggest_strikes(surface, spot))
    m["_prepared"] = prepared
    m["_surface"] = surface
    return m


def scan(tickers=None, asof=None, make_plots=False, client=None):
    tickers = tickers or config.UNIVERSE
    asof = asof or str(date.today())
    asof_ts = pd.Timestamp(asof)
    if client is None:
        client = PolygonClient(cache_date=asof)
    history = store.load_history()

    rows = []
    for ticker in tickers:
        try:
            rows.append(_process_ticker(client, ticker, asof_ts, history))
            print(f"ok   {ticker}")
        except Exception as e:  # noqa: BLE001 - one bad ticker must not kill scan
            print(f"skip {ticker}: {e}")

    plain = [{k: v for k, v in r.items() if not k.startswith("_")}
             for r in rows]
    if plain:
        store.save_metrics(plain, asof)
    wheel, leaps = signals.build_report(plain)

    pd.set_option("display.width", 200)
    print(f"\n=== Wheel / short-put candidates (<={config.SHORT_DTE_MAX} DTE, "
          f"sell ~25d puts) --- {asof} ===")
    print(wheel.round(3).to_string(index=False) if not wheel.empty
          else "(none)")
    print(f"\n=== LEAPS candidates (>= {config.LEAPS_T_MIN}y, "
          f"buy ~75d calls) --- {asof} ===")
    print(leaps.round(3).to_string(index=False) if not leaps.empty
          else "(none)")
    print("\nReminder: check earnings dates before selling short-dated puts.")

    if make_plots:
        for r in rows:
            paths = plots.plot_ticker(r["_prepared"], r["_surface"],
                                      r["underlying"], config.PLOTS_DIR)
            print(f"plots: {', '.join(str(p) for p in paths)}")
    return wheel, leaps


def plot_one(ticker: str, asof=None):
    asof = asof or str(date.today())
    client = PolygonClient(cache_date=asof)
    asof_ts = pd.Timestamp(asof)
    chain_df = chain.fetch_chain(client, ticker)
    prepared = clean.prepare(chain_df, asof=asof_ts)
    surface = ssvi.fit_surface(prepared)
    for p in plots.plot_ticker(prepared, surface, ticker, config.PLOTS_DIR):
        print(p)


def main():
    parser = argparse.ArgumentParser(prog="ssvi")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_scan = sub.add_parser("scan", help="daily scan of the universe")
    p_scan.add_argument("--plots", action="store_true")
    p_scan.add_argument("--tickers", type=str, default=None,
                        help="comma-separated override, e.g. NVDA,AAPL")
    p_plot = sub.add_parser("plot", help="plot one ticker from today's cache")
    p_plot.add_argument("ticker")
    args = parser.parse_args()
    if args.cmd == "scan":
        tickers = args.tickers.split(",") if args.tickers else None
        scan(tickers=tickers, make_plots=args.plots)
    elif args.cmd == "plot":
        plot_one(args.ticker)


if __name__ == "__main__":
    main()
