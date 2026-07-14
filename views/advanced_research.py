"""
Investigación avanzada — laboratorio técnico (V6, V14, V17, backtests).
Accesible solo desde enlace en «Cómo funciona».
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

from ui.styles import apply_app_styles

from src.data import download_price_data
from src.indicators import add_indicators
from src.strategy_backtest import analyze_ticker
from src.portfolio_v6 import (
    get_current_v6_portfolio_signal,
    get_required_tickers,
    load_v6_config,
)
from src.portfolio_v14 import (
    get_current_v14_portfolio_signal,
    get_required_tickers as get_required_tickers_v14,
    load_v14_config,
)

apply_app_styles()

st.markdown("# Investigación avanzada")
st.caption("Laboratorio técnico: V6, V14, V17, backtests, gráficas históricas y modos experimentales.")

DEFAULT_TICKERS = "SPY, QQQ, AAPL, MSFT, NVDA, TSLA, AMD, META, GOOGL, AMZN"
PERIOD_SWING = {"6 meses": "6mo", "1 año": "1y", "2 años": "2y", "5 años": "5y"}
PERIOD_INTRADAY = {"5 días": "5d", "1 mes": "1mo", "60 días": "60d"}
HOLDING_SWING = {"1 día": 1, "2 días": 2, "3 días": 3, "5 días": 5}
HOLDING_INTRADAY = {"3 velas": 3, "5 velas": 5, "8 velas": 8, "13 velas": 13}

SIGNAL_LABEL = {
    "BUY": "Buscar compra",
    "SELL": "Salir / no comprar",
    "HOLD": "Esperar",
    "AVOID": "Evitar",
}
SIGNAL_COLOR = {
    "BUY": "#22c55e",
    "SELL": "#ef4444",
    "HOLD": "#3b82f6",
    "AVOID": "#9ca3af",
}
RISK_LABEL = {"Low": "Bajo", "Medium": "Medio", "High": "Alto"}
HOLD_SCORE_MIN = 45

ACTION_TEXT = {
    "BUY": "El algoritmo permite buscar compra. Entrada: próxima apertura o ruptura del máximo reciente. Salida: si aparece SELL o al cumplirse la duración máxima.",
    "SELL": "Si ya estás dentro, la señal sugiere salir o reducir. Si no estás dentro, no compraría ahora.",
    "HOLD": "No hay entrada clara. Mejor esperar.",
    "AVOID": "Demasiado riesgo o backtest débil. Mejor no tocar.",
}

SELL_NOTE = (
    "SELL no significa abrir una operación en corto. Significa que el algoritmo ve debilidad. "
    "Si ya tuvieras este activo, sugiere salir o reducir. Si no lo tienes, sugiere no comprar ahora."
)

ASSET_DESCRIPTIONS = {
    "SPY": "ETF que sigue al S&P 500. Representa unas 500 grandes empresas de Estados Unidos.",
    "QQQ": "ETF del Nasdaq 100. Mucho peso en tecnológicas como Apple, Microsoft, Nvidia, Amazon, Meta y Google.",
    "AAPL": "Apple. Empresa tecnológica conocida por iPhone, Mac, servicios y dispositivos.",
    "MSFT": "Microsoft. Software, nube, Windows, Office, Azure e inteligencia artificial.",
    "NVDA": "Nvidia. Chips gráficos e inteligencia artificial.",
    "TSLA": "Tesla. Coches eléctricos, baterías y tecnología.",
    "AMD": "AMD. Procesadores y chips gráficos.",
    "META": "Meta. Facebook, Instagram, WhatsApp y proyectos de IA.",
    "GOOGL": "Alphabet / Google. Buscador, YouTube, nube, publicidad e IA.",
    "AMZN": "Amazon. Ecommerce, nube AWS, logística y Prime.",
}


def get_asset_description(ticker: str) -> str:
    return ASSET_DESCRIPTIONS.get(
        ticker.upper(),
        "Activo financiero. Revisa qué empresa o ETF es antes de tomar decisiones.",
    )


def score_label(signal: str) -> str:
    if signal in ("BUY", "HOLD"):
        return "Calidad del setup"
    if signal == "SELL":
        return "Confianza de debilidad"
    return "Riesgo / baja calidad"


def inject_css():
    st.markdown("""
    <style>
        .stApp { background: #0f172a; }
        [data-testid="stSidebar"] { background: #111827; }
        .card {
            background: #1f2937; border: 1px solid #374151;
            border-radius: 18px; padding: 1.5rem 1.8rem; margin-bottom: 1.2rem;
        }
        .card-tag { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 2px; color: #9ca3af; margin-bottom: 0.8rem; }
        .section-title { font-size: 1rem; font-weight: 700; color: #e5e7eb; margin: 1.8rem 0 0.6rem; }
        .section-hint { font-size: 0.82rem; color: #9ca3af; margin-bottom: 0.8rem; }
        .ticker-big { font-size: 2.2rem; font-weight: 800; color: #f9fafb; line-height: 1.1; }
        .asset-desc { font-size: 0.92rem; color: #cbd5e1; margin: 0.4rem 0 1rem; line-height: 1.4; }
        .signal-big { font-size: 3rem; font-weight: 900; line-height: 1; margin: 0.2rem 0; }
        .signal-sub { font-size: 1.1rem; color: #d1d5db; margin-bottom: 1rem; }
        .meta-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 1rem; }
        .meta-item label { font-size: 0.68rem; color: #9ca3af; text-transform: uppercase; display: block; }
        .meta-item span { font-size: 0.95rem; font-weight: 600; color: #f3f4f6; }
        .mini-stat {
            background: #111827; border: 1px solid #374151; border-radius: 14px;
            padding: 1rem; text-align: center;
        }
        .mini-stat-num { font-size: 1.8rem; font-weight: 800; color: #f9fafb; }
        .mini-stat-lbl { font-size: 0.8rem; color: #9ca3af; margin-top: 0.2rem; }
        .sig-card {
            background: #111827; border: 1px solid #374151; border-radius: 14px;
            padding: 1rem; margin-bottom: 0.6rem; height: 100%;
        }
        .sig-ticker { font-size: 1.05rem; font-weight: 700; color: #f9fafb; }
        .sig-desc { font-size: 0.78rem; color: #9ca3af; margin: 0.25rem 0 0.5rem; line-height: 1.35; }
        .cmp-card {
            background: #111827; border: 1px solid #374151; border-radius: 14px;
            padding: 1rem; margin-bottom: 0.6rem;
        }
        .cmp-name { font-weight: 700; color: #f3f4f6; }
        .cmp-row { font-size: 0.85rem; color: #cbd5e1; margin-top: 0.3rem; }
        .beat { color: #22c55e; } .lose { color: #ef4444; }
        .title { font-size: 2rem; font-weight: 800; color: #f9fafb; margin: 0; }
        .subtitle { color: #9ca3af; font-size: 0.95rem; margin-top: 0.2rem; }
        .empty-msg { color: #9ca3af; font-size: 0.9rem; padding: 0.5rem 0 1rem; }
        .sell-note { background: #450a0a; border: 1px solid #991b1b; border-radius: 10px;
            padding: 0.7rem 1rem; color: #fca5a5; font-size: 0.85rem; margin-top: 0.8rem; }
        .v6-banner {
            background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
            border: 2px solid #ef4444; border-radius: 14px; padding: 1rem 1.2rem;
            margin-bottom: 1rem; color: #fecaca; font-weight: 700; text-align: center;
        }
        .v6-ok-banner {
            background: #14532d; border: 1px solid #22c55e; border-radius: 12px;
            padding: 0.8rem 1rem; color: #bbf7d0; margin-bottom: 0.8rem;
        }
        .v6-metric { background: #111827; border: 1px solid #374151; border-radius: 12px;
            padding: 1rem; text-align: center; }
        .v6-metric-num { font-size: 1.5rem; font-weight: 800; color: #f9fafb; }
        .v6-metric-lbl { font-size: 0.75rem; color: #9ca3af; margin-top: 0.25rem; }
        .weight-row { display: flex; justify-content: space-between; padding: 0.35rem 0;
            border-bottom: 1px solid #374151; color: #e5e7eb; font-size: 0.92rem; }
        #MainMenu, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


def parse_tickers(text: str) -> list[str]:
    return [t.strip().upper() for t in text.replace("\n", ",").split(",") if t.strip()]


def fmt_pct(v, default="No disponible") -> str:
    if v is None:
        return default
    try:
        return f"{float(v):+.1f}%"
    except (TypeError, ValueError):
        return default


def fmt_price(v, default="No disponible") -> str:
    if v is None:
        return default
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return default


def fmt_num(v, default="No disponible") -> str:
    if v is None:
        return default
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return default


def horizon_label(is_swing: bool, holding: int) -> str:
    if is_swing:
        return f"máximo {holding} día{'s' if holding > 1 else ''}"
    return f"máximo {holding} velas"


def short_strategy(name: str) -> str:
    if not name:
        return "No disponible"
    return name.replace("Weakness / Exit Signal", "Weakness Signal")


def resolve_dataframe(primary, fallback=None):
    df = primary
    if df is None or getattr(df, "empty", True):
        df = fallback
    if df is None or getattr(df, "empty", True):
        return None
    return df


def safe_excess(metrics: dict) -> float:
    v = metrics.get("excess_return")
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def run_analysis(tickers, mode_key, period_key, interval, holding, progress_bar):
    results, errors = [], []
    for i, ticker in enumerate(tickers):
        progress_bar.progress((i + 1) / len(tickers), text=f"Analizando {ticker}...")
        try:
            df = download_price_data(ticker, period=period_key, interval=interval)
            df = add_indicators(df)
            analysis = analyze_ticker(ticker, df, mode_key, holding)
            br = analysis.get("best", {})
            results.append({
                "ticker": ticker.upper(),
                "signal": br.get("current_signal", "HOLD"),
                "score": br.get("score", 0),
                "risk": br.get("risk_level", "Medium"),
                "strategy": br.get("strategy_name", "No disponible"),
                "metrics": br.get("metrics", {}),
                "avoid_reason": br.get("avoid_reason", ""),
                "analysis": analysis,
            })
        except Exception as e:
            errors.append(f"{ticker}: {e}")
    return results, errors


def categorize_results(results: list) -> tuple[list, list, list, list]:
    """Separa resultados en BUY, HOLD, SELL y AVOID."""
    buy = sorted([r for r in results if r.get("signal") == "BUY"], key=lambda x: x.get("score", 0), reverse=True)
    hold = sorted([r for r in results if r.get("signal") == "HOLD"], key=lambda x: x.get("score", 0), reverse=True)
    sell = sorted([r for r in results if r.get("signal") == "SELL"], key=lambda x: x.get("score", 0), reverse=True)
    avoid = sorted([r for r in results if r.get("signal") == "AVOID"], key=lambda x: x.get("score", 0), reverse=True)
    return buy, hold, sell, avoid


def pick_main_feature(buy: list, hold: list, sell: list) -> tuple[str, str, dict | None]:
    """
    Elige la tarjeta principal. SELL nunca es 'mejor oportunidad'.
    Returns: (mode, title, item_or_none)
    """
    if buy:
        return "buy", "Mejor oportunidad de compra", buy[0]
    if hold and hold[0].get("score", 0) >= HOLD_SCORE_MIN:
        return "hold", "Activo en vigilancia", hold[0]
    if sell:
        return "sell_only", "No hay compras claras. Hay alertas de debilidad.", None
    return "none", "No hay oportunidades claras ahora", None


def build_analysis_map(results: list) -> dict:
    return {r.get("ticker", ""): r.get("analysis", {}) for r in results if r.get("ticker")}


def render_summary_counts(buy: list, hold: list, sell: list, avoid: list):
    cols = st.columns(4)
    for col, label, count, color in zip(
        cols,
        ["Compras", "Vigilancia", "Alertas", "Evitar"],
        [len(buy), len(hold), len(sell), len(avoid)],
        ["#22c55e", "#3b82f6", "#ef4444", "#9ca3af"],
    ):
        with col:
            st.markdown(f"""
            <div class="mini-stat">
                <div class="mini-stat-num" style="color:{color};">{count}</div>
                <div class="mini-stat-lbl">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def render_main_card(title: str, item: dict | None, is_swing: bool, holding: int, mode: str):
    if item is None:
        st.markdown(f"""
        <div class="card">
            <div class="card-tag">{title}</div>
            <p style="color:#cbd5e1;font-size:1rem;margin:0.5rem 0;">Revisa las secciones de abajo para más detalle.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    signal = item.get("signal", "HOLD")
    color = SIGNAL_COLOR.get(signal, "#9ca3af")
    ticker = item.get("ticker", "—")
    desc = get_asset_description(ticker)
    score = item.get("score", 0)
    analysis = item.get("analysis", {})
    slabel = score_label(signal)
    stop = fmt_price(analysis.get("stop_price")) if signal == "BUY" else "No aplica"

    extra = ""
    if signal == "AVOID" and item.get("avoid_reason"):
        extra = f'<p style="color:#fbbf24;font-size:0.85rem;margin-top:0.8rem;">⚠️ {item["avoid_reason"]}</p>'

    st.markdown(f"""
    <div class="card">
        <div class="card-tag">{title}</div>
        <div class="ticker-big">{ticker}</div>
        <div class="asset-desc">{desc}</div>
        <div class="signal-big" style="color:{color};">{signal}</div>
        <div class="signal-sub">{SIGNAL_LABEL.get(signal, signal)}</div>
        <div class="meta-grid">
            <div class="meta-item"><label>{slabel}</label><span>{score}/100</span></div>
            <div class="meta-item"><label>Riesgo</label><span>{RISK_LABEL.get(item.get('risk','Medium'), 'Medio')}</span></div>
            <div class="meta-item"><label>Estrategia</label><span>{short_strategy(item.get('strategy',''))}</span></div>
            <div class="meta-item"><label>Duración máx.</label><span>{horizon_label(is_swing, holding)}</span></div>
            <div class="meta-item"><label>Último precio</label><span>{fmt_price(analysis.get('last_price'))}</span></div>
            <div class="meta-item"><label>Stop orientativo</label><span>{stop}</span></div>
        </div>
        {extra}
    </div>
    """, unsafe_allow_html=True)


def render_action_card(item: dict | None, mode: str):
    if item is None:
        if mode == "sell_only":
            st.markdown(f"""
            <div class="card">
                <div class="card-tag">Qué hago con esto</div>
                <div style="color:#cbd5e1;">No hay compras recomendadas. Mira la sección <strong>Alertas de salida</strong> abajo.</div>
                <div class="sell-note">{SELL_NOTE}</div>
            </div>
            """, unsafe_allow_html=True)
        return

    signal = item.get("signal", "HOLD")
    text = ACTION_TEXT.get(signal, "")
    sell_html = f'<div class="sell-note">{SELL_NOTE}</div>' if signal == "SELL" else ""
    st.markdown(f"""
    <div class="card">
        <div class="card-tag">Qué hago con esto</div>
        <div style="color:#cbd5e1;line-height:1.6;">{text}</div>
        {sell_html}
    </div>
    """, unsafe_allow_html=True)


def render_signal_section(title: str, hint: str, items: list, empty_msg: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if hint:
        st.markdown(f'<div class="section-hint">{hint}</div>', unsafe_allow_html=True)

    if not items:
        st.markdown(f'<div class="empty-msg">{empty_msg}</div>', unsafe_allow_html=True)
        return

    ncols = min(len(items), 3) if len(items) <= 6 else 4
    cols = st.columns(ncols)
    for i, r in enumerate(items):
        sig = r.get("signal", "HOLD")
        color = SIGNAL_COLOR.get(sig, "#9ca3af")
        m = r.get("metrics", {})
        ticker = r.get("ticker", "—")
        desc = get_asset_description(ticker)
        if len(desc) > 70:
            desc = desc[:70] + "…"
        with cols[i % ncols]:
            st.markdown(f"""
            <div class="sig-card">
                <div class="sig-ticker">{ticker}</div>
                <div class="sig-desc">{desc}</div>
                <div style="color:{color};font-weight:700;">{sig}</div>
                <div style="font-size:0.8rem;color:#9ca3af;margin-top:0.35rem;">
                    {score_label(sig)}: {r.get('score', 0)}/100<br>
                    {short_strategy(r.get('strategy',''))}<br>
                    Resultado {fmt_pct(m.get('strategy_total_return'))} vs {fmt_pct(m.get('buy_hold_total_return'))}
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_chart(analysis: dict, ticker: str):
    br = analysis.get("best", {})
    df_bt = resolve_dataframe(br.get("df_backtest"), analysis.get("df"))
    if df_bt is None:
        st.info("Gráfico no disponible.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt["Close"], name="Precio", line=dict(color="#60a5fa", width=2)))
    for col, name, c in [("EMA_9", "EMA 9", "#fbbf24"), ("EMA_21", "EMA 21", "#34d399"), ("SMA_50", "SMA 50", "#a78bfa")]:
        if col in df_bt.columns and df_bt[col].notna().any():
            fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt[col], name=name, line=dict(color=c, width=1.2)))
    if "Buy_Signal" in df_bt.columns:
        buys = df_bt[df_bt["Buy_Signal"]]
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers", name="Compra",
                marker=dict(color="#22c55e", size=9, symbol="triangle-up")))
    if "Sell_Signal" in df_bt.columns:
        sells = df_bt[df_bt["Sell_Signal"]]
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers", name="Salida",
                marker=dict(color="#ef4444", size=9, symbol="triangle-down")))
    fig.update_layout(
        title=f"{ticker} — {short_strategy(br.get('strategy_name', ''))}",
        height=480, hovermode="x unified", plot_bgcolor="#1f2937", paper_bgcolor="#0f172a",
        font=dict(color="#e5e7eb"), legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor="#374151")
    fig.update_yaxes(gridcolor="#374151")
    st.plotly_chart(fig, use_container_width=True)


def render_market_comparison(results_list: list):
    tradable = [r for r in results_list if r.get("strategy_name") != "Weakness / Exit Signal"]
    for r in sorted(tradable, key=lambda x: x.get("score", 0), reverse=True):
        m = r.get("metrics", {})
        excess = safe_excess(m)
        cls = "beat" if excess > 0 else "lose"
        st.markdown(f"""
        <div class="cmp-card">
            <div class="cmp-name">{r.get('strategy_name', '—')}</div>
            <div class="cmp-row">
                Resultado estrategia <strong>{fmt_pct(m.get('strategy_total_return'))}</strong> ·
                Comprar y mantener <strong>{fmt_pct(m.get('buy_hold_total_return'))}</strong> ·
                Diferencia <span class="{cls}">{fmt_pct(excess)}</span>
            </div>
            <div class="cmp-row">
                Peor caída {fmt_pct(m.get('max_drawdown'))} ·
                Ops {fmt_num(m.get('num_trades'))} ·
                Acierto {fmt_pct(m.get('win_rate'))}
            </div>
        </div>
        """, unsafe_allow_html=True)


def download_v6_data_dict(tickers: list[str], period: str = "5y") -> tuple[dict, list[str]]:
    """Descarga datos diarios para tickers V6. Devuelve data_dict y errores."""
    data_dict, errors = {}, []
    for ticker in tickers:
        t = ticker.strip().upper()
        if not t or t == "CASH":
            continue
        try:
            data_dict[t] = download_price_data(t, period=period, interval="1d")
        except Exception as exc:
            errors.append(f"{t}: {exc}")
    return data_dict, errors


def render_v6_equity_chart(strategy_key: str = "blended_champion_weights_alpha_0.5"):
    eq_path = Path("research_outputs/v6/research_v6_equity_curves.csv")
    if not eq_path.exists():
        return False
    try:
        eq_df = pd.read_csv(eq_path, index_col=0, parse_dates=True)
    except Exception:
        return False
    if eq_df.empty:
        return False

    fig = go.Figure()
    palette = {
        strategy_key: "#22c55e",
        "champion_trend_following_v4": "#60a5fa",
        "SPY": "#fbbf24",
        "QQQ": "#a78bfa",
        "EW": "#94a3b8",
    }
    for col in eq_df.columns:
        if col == strategy_key or col in ("SPY", "QQQ", "EW", "champion_trend_following_v4"):
            fig.add_trace(go.Scatter(
                x=eq_df.index, y=eq_df[col], name=col,
                line=dict(color=palette.get(col, "#cbd5e1"), width=2 if col == strategy_key else 1.2),
            ))
    fig.update_layout(
        title="Curva de equity V6 (backtest investigación)",
        height=420, hovermode="x unified", plot_bgcolor="#1f2937", paper_bgcolor="#0f172a",
        font=dict(color="#e5e7eb"), legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor="#374151")
    fig.update_yaxes(gridcolor="#374151")
    st.plotly_chart(fig, use_container_width=True)
    return True


def render_v14_equity_chart(data_dict: dict | None = None):
    eq_path = Path("research_outputs/v14/research_v14_equity_curve.csv")
    if not eq_path.exists():
        return False

    eq_df = pd.read_csv(eq_path, index_col=0, parse_dates=True)
    if eq_df.empty or "equity" not in eq_df.columns:
        return False

    v14 = eq_df["equity"].dropna()
    v14_norm = v14 / v14.iloc[0] * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v14_norm.index, y=v14_norm.values, name="V14 R1", line=dict(color="#22c55e", width=2)))

    if data_dict:
        for ticker, color in [("SPY", "#3b82f6"), ("QQQ", "#a855f7")]:
            if ticker not in data_dict:
                continue
            df = data_dict[ticker]
            if df is None or df.empty or "Close" not in df.columns:
                continue
            px = df["Close"].reindex(v14_norm.index).ffill().dropna()
            if len(px) < 2:
                continue
            norm = px / px.iloc[0] * 100
            fig.add_trace(go.Scatter(x=norm.index, y=norm.values, name=ticker, line=dict(color=color, width=1.5)))

        if "SPY" in data_dict and "TLT" in data_dict:
            spy = data_dict["SPY"]["Close"].reindex(v14_norm.index).ffill()
            tlt = data_dict["TLT"]["Close"].reindex(v14_norm.index).ffill()
            mix = 0.6 * spy + 0.4 * tlt
            mix = mix.dropna()
            if len(mix) >= 2:
                mix_norm = mix / mix.iloc[0] * 100
                fig.add_trace(go.Scatter(
                    x=mix_norm.index, y=mix_norm.values, name="60/40",
                    line=dict(color="#f59e0b", width=1.5, dash="dot"),
                ))

    fig.update_layout(
        title="Curva de equity V14 vs benchmarks (normalizado base 100)",
        height=420,
        plot_bgcolor="#1f2937",
        paper_bgcolor="#0f172a",
        font=dict(color="#e5e7eb"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(gridcolor="#374151")
    fig.update_yaxes(gridcolor="#374151")
    st.plotly_chart(fig, use_container_width=True)
    return True


def render_v14_mode(capital: float, universe_text: str, calc_v14: bool):
    try:
        v14_config = load_v14_config()
    except Exception as exc:
        st.error(f"No se pudo cargar config V14: {exc}")
        return

    default_universe = ", ".join(get_required_tickers_v14(v14_config))
    tickers = parse_tickers(universe_text) if universe_text.strip() else get_required_tickers_v14(v14_config)
    if not tickers:
        tickers = get_required_tickers_v14(v14_config)

    st.markdown('<p class="title">📊 Portfolio V14 Approved Paper Trading</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">V14 R1 Return Engine — momentum semanal A_tsmom_63. Solo simulación, sin órdenes reales.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="v6-banner">
        ⚠️ PAPER TRADING · NO DINERO REAL · Backtest no garantiza resultados futuros
    </div>
    """, unsafe_allow_html=True)

    if not calc_v14:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:2.5rem;">
            <div style="font-size:2.5rem;">🧪</div>
            <p style="color:#9ca3af;">
                Pulsa <strong>Calcular cartera V14</strong> en el panel izquierdo.<br>
                Universo por defecto: <code>{default_universe[:80]}…</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    with st.spinner("Descargando datos y calculando cartera V14..."):
        data_dict, dl_errors = download_v6_data_dict(tickers, period="5y")
        signal = get_current_v14_portfolio_signal(data_dict, capital=capital, config=v14_config)

    if signal.get("error"):
        st.error(signal["error"])
        if dl_errors:
            with st.expander("Errores de descarga"):
                for e in dl_errors:
                    st.text(e)
        return

    bt = signal.get("backtest_summary", {})
    target_weights = signal.get("target_weights", {})
    alloc = signal.get("capital_allocation", {})
    signals_df = signal.get("signals", pd.DataFrame())
    risk_mode = signal.get("risk_mode", "")

    st.markdown(f"""
    <div class="card">
        <div class="card-tag">Portfolio V14 Approved Paper Trading</div>
        <div class="ticker-big">{signal.get('strategy_name', 'V14 R1 Return Engine')}</div>
        <div class="v6-ok-banner">
            Estado: APPROVED FOR WEB PAPER · Dinero real: NO ·
            Score: {signal.get('score', 85)}/100 · Rebalanceo: semanal (viernes)
        </div>
        <div class="meta-grid">
            <div class="meta-item"><label>Última fecha datos</label><span>{signal.get('last_date', 'N/A')}</span></div>
            <div class="meta-item"><label>Rebalanceo señal</label><span>{signal.get('rebalance_date', 'N/A')}</span></div>
            <div class="meta-item"><label>Capital simulado</label><span>${capital:,.0f}</span></div>
            <div class="meta-item"><label>Modo riesgo</label><span>{risk_mode}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Métricas de investigación V14</div>', unsafe_allow_html=True)
    mcols = st.columns(6)
    metrics = [
        ("CAGR", f"{bt.get('CAGR', 0):.2f}%"),
        ("Sharpe", f"{bt.get('sharpe', 0):.3f}"),
        ("Sortino", f"{bt.get('sortino', 0):.3f}"),
        ("Max DD", f"{bt.get('max_drawdown', 0):.2f}%"),
        ("Robustez", f"{bt.get('robustness_score', 98)}"),
        ("Overfitting", f"{bt.get('overfitting_risk', 'LOW')}"),
    ]
    for col, (lbl, val) in zip(mcols, metrics):
        with col:
            st.markdown(f"""
            <div class="v6-metric">
                <div class="v6-metric-num">{val}</div>
                <div class="v6-metric-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Qué está haciendo ahora</div>', unsafe_allow_html=True)
    buy_count = len(signals_df[signals_df["signal"] == "BUY"]) if len(signals_df) else 0
    shy_pct = target_weights.get("SHY", 0) + target_weights.get("CASH", 0)
    if risk_mode == "defensive_100_shy" or shy_pct >= 0.99:
        st.info(
            "El modelo está defensivo. No hay BUY actuales. "
            "Mantiene SHY/CASH hasta la próxima revisión semanal."
        )
    elif buy_count == 0:
        st.info(
            "No hay nuevas compras (BUY) esta semana. "
            "La cartera mantiene posiciones actuales hasta el próximo rebalanceo."
        )
    else:
        st.success(f"Hay {buy_count} señal(es) BUY en esta revisión.")

    st.markdown('<div class="section-title">Señales y asignación</div>', unsafe_allow_html=True)
    if len(signals_df):
        display_df = signals_df.copy()
        display_df["simulated_amount"] = display_df.apply(
            lambda r: alloc.get(r["ticker"], round(capital * r["target_weight"], 2)), axis=1
        )

        st.dataframe(
            display_df[["ticker", "signal", "target_weight", "simulated_amount", "reason"]].rename(columns={
                "ticker": "Ticker",
                "signal": "Señal",
                "target_weight": "Peso objetivo",
                "simulated_amount": "Importe simulado",
                "reason": "Motivo",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Sin posiciones activas en cartera.")

    st.markdown('<div class="section-title">Comparación V14 vs V6</div>', unsafe_allow_html=True)
    try:
        v6_cfg = load_v6_config()
        v6_bt = v6_cfg.get("backtest_summary", {})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="cmp-card">
                <div class="cmp-name">V14 R1 Return Engine</div>
                <div class="cmp-row">CAGR: {bt.get('CAGR', 0):.2f}% · Sharpe: {bt.get('sharpe', 0):.3f}</div>
                <div class="cmp-row">Max drawdown: {bt.get('max_drawdown', 0):.2f}%</div>
                <div class="cmp-row">Win años vs SPY: {bt.get('win_years_vs_spy', 0)*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="cmp-card">
                <div class="cmp-name">V6 Blended Champion (histórico)</div>
                <div class="cmp-row">CAGR: {v6_bt.get('CAGR', 0):.2f}% · Sharpe: {v6_bt.get('sharpe', 0):.3f}</div>
                <div class="cmp-row">Max drawdown: {v6_bt.get('max_drawdown', 0):.2f}%</div>
                <div class="cmp-row">Champion histórico del proyecto</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <div class="sell-note" style="margin-top:0.5rem;">
            V14 tiene menor drawdown histórico simulado que V6 (-23.88% vs -34.91%).
            V6 tuvo más rentabilidad histórica total, pero con caídas mayores.
            V14 es más conservadora y robusta para paper trading experimental.
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.caption("Comparación V6 no disponible (config V6 no encontrada).")

    st.markdown('<div class="section-title">Gráfico</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if not render_v14_equity_chart(data_dict):
        st.info(
            "Curva V14 no disponible. Ejecuta el notebook V14 y copia "
            "research_v14_equity_curve.csv a research_outputs/v14/."
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Advertencias</div>', unsafe_allow_html=True)
    for w in signal.get("warnings", []):
        st.warning(w)

    with st.expander("Ver detalles técnicos"):
        st.json({
            "target_weights": target_weights,
            "capital_allocation": alloc,
            "risk_mode": risk_mode,
            "base_engine": v14_config.get("base_engine"),
        })
        st.markdown("**Config completa**")
        st.json(v14_config)

    if dl_errors:
        with st.expander(f"Tickers no descargados ({len(dl_errors)})"):
            for e in dl_errors:
                st.text(e)


def render_v6_mode(capital: float, universe_text: str, calc_v6: bool):
    try:
        v6_config = load_v6_config()
    except Exception as exc:
        st.error(f"No se pudo cargar config V6: {exc}")
        return

    default_universe = ", ".join(get_required_tickers(v6_config))
    tickers = parse_tickers(universe_text) if universe_text.strip() else get_required_tickers(v6_config)
    if not tickers:
        tickers = get_required_tickers(v6_config)

    st.markdown('<p class="title">📊 Portfolio V6 Paper Trading</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Estrategia oficial de investigación V6 — solo simulación, sin órdenes reales.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="v6-banner">
        ⚠️ NO DINERO REAL · PAPER TRADING · Backtest no garantiza resultados futuros
    </div>
    """, unsafe_allow_html=True)

    if not calc_v6:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:2.5rem;">
            <div style="font-size:2.5rem;">🧪</div>
            <p style="color:#9ca3af;">
                Pulsa <strong>Calcular cartera V6</strong> en el panel izquierdo.<br>
                Universo por defecto: <code>{default_universe[:80]}…</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    with st.spinner("Descargando datos y calculando cartera V6..."):
        data_dict, dl_errors = download_v6_data_dict(tickers)
        signal = get_current_v6_portfolio_signal(data_dict, capital=capital, config=v6_config)

    if signal.get("error"):
        st.error(signal["error"])
        if dl_errors:
            with st.expander("Errores de descarga"):
                for e in dl_errors:
                    st.text(e)
        return

    bt = signal.get("backtest_summary", {})
    weights = signal.get("weights", {})
    alloc = signal.get("capital_allocation", {})
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    st.markdown(f"""
    <div class="card">
        <div class="card-tag">Portfolio V6 Paper Trading</div>
        <div class="ticker-big">{signal.get('strategy_name', 'Blended Champion V6')}</div>
        <div class="v6-ok-banner">
            Estado: APROBADA SOLO PARA PAPER TRADING · Dinero real: NO ·
            Score investigación: {signal.get('final_score_v6', 100)}/100
        </div>
        <div class="meta-grid">
            <div class="meta-item"><label>Última fecha</label><span>{signal.get('last_date', 'N/A')}</span></div>
            <div class="meta-item"><label>Blend</label><span>50% V4 + 50% Adaptive</span></div>
            <div class="meta-item"><label>Capital simulado</label><span>${capital:,.0f}</span></div>
            <div class="meta-item"><label>Pesos activos</label><span>{sum(1 for _, w in weights.items() if w > 0.001)}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Asignación sugerida</div>', unsafe_allow_html=True)
    weight_html = "".join(
        f'<div class="weight-row"><span>{t}</span><strong>{w*100:.1f}%</strong></div>'
        for t, w in sorted_weights if w > 0.001
    )
    alloc_html = "".join(
        f'<div class="weight-row"><span>{t}</span><strong>${amt:,.0f}</strong></div>'
        for t, amt in sorted(alloc.items(), key=lambda x: x[1], reverse=True) if amt > 0
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="card"><div class="card-tag">Pesos objetivo</div>{weight_html}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="card"><div class="card-tag">Capital simulado: ${capital:,.0f}</div>{alloc_html}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f"""
    <div class="card">
        <div class="card-tag">Qué hace este algoritmo</div>
        <p style="color:#cbd5e1;line-height:1.6;margin:0;">
            Combina una estrategia de tendencia V4 con un modelo adaptativo.
            Mantiene activos fuertes cuando el mercado acompaña y aumenta defensivos
            cuando el mercado se debilita.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Resumen de investigación V6</div>', unsafe_allow_html=True)
    mcols = st.columns(6)
    metrics = [
        ("CAGR", f"{bt.get('CAGR', 0):.2f}%"),
        ("Sharpe", f"{bt.get('sharpe', 0):.3f}"),
        ("Sortino", f"{bt.get('sortino', 0):.3f}"),
        ("Peor caída", f"{bt.get('max_drawdown', 0):.2f}%"),
        ("vs SPY", f"+{bt.get('excess_vs_spy', 0):.2f}%"),
        ("vs QQQ", f"+{bt.get('excess_vs_qqq', 0):.2f}%"),
    ]
    for col, (lbl, val) in zip(mcols, metrics):
        with col:
            st.markdown(f"""
            <div class="v6-metric">
                <div class="v6-metric-num">{val}</div>
                <div class="v6-metric-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sell-note" style="margin-top:1rem;">
        {signal.get('risk_message', '')}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Gráfico</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if not render_v6_equity_chart(v6_config.get("selected_strategy", "blended_champion_weights_alpha_0.5")):
        main_tickers = [t for t in ["SPY", "QQQ", "NVDA"] if t in data_dict]
        if main_tickers:
            fig = go.Figure()
            for t in main_tickers:
                df = data_dict[t]
                fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name=t))
            fig.update_layout(
                title="Precios recientes (equity curve V6 no disponible localmente)",
                height=400, plot_bgcolor="#1f2937", paper_bgcolor="#0f172a", font=dict(color="#e5e7eb"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Gráfico no disponible. Exporta research_v6_equity_curves.csv a research_outputs/v6/.")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Ver detalles técnicos"):
        st.json({
            "weights": weights,
            "capital_allocation": alloc,
            "warnings": signal.get("warnings", []),
            "config_summary": {
                "strategy_id": v6_config.get("strategy_id"),
                "blend": v6_config.get("blend"),
                "parameters": v6_config.get("parameters"),
            },
        })
        st.markdown("**Config completa**")
        st.json(v6_config)

    if dl_errors:
        with st.expander(f"Tickers no descargados ({len(dl_errors)})"):
            for e in dl_errors:
                st.text(e)


# ── UI ──────────────────────────────────────────────────────────────────────
inject_css()

with st.sidebar:
    st.markdown("### Modo")
    app_mode = st.radio(
        "Selecciona modo",
        [
            "Portfolio V14 Approved Paper Trading",
            "Portfolio V6 Paper Trading",
            "Swing 1-5 días",
            "Intradía experimental",
        ],
        index=0,
        label_visibility="collapsed",
    )

    if app_mode == "Portfolio V14 Approved Paper Trading":
        st.markdown("#### Portfolio V14")
        st.caption("Nueva estrategia aprobada — A_tsmom_63 paper trading.")
        try:
            _v14_cfg = load_v14_config()
            default_v14_tickers = ", ".join(get_required_tickers_v14(_v14_cfg))
        except Exception:
            default_v14_tickers = DEFAULT_TICKERS + ", SHY, IEF, TLT, GLD, MTUM, USMV"
        v14_capital = st.number_input("Capital simulado", min_value=1000, value=10000, step=1000, key="v14_capital")
        v14_universe = st.text_area("Universo (tickers)", value=default_v14_tickers, height=100, key="v14_universe")
        calc_v14 = st.button("Calcular cartera V14", type="primary", use_container_width=True)
        st.caption("Paper trading experimental. No ejecuta operaciones.")
        st.caption("No es asesoramiento financiero.")
    elif app_mode == "Portfolio V6 Paper Trading":
        st.markdown("#### Portfolio V6")
        st.caption("Modo recomendado para paper trading experimental.")
        try:
            _v6_cfg = load_v6_config()
            default_v6_tickers = ", ".join(get_required_tickers(_v6_cfg))
        except Exception:
            default_v6_tickers = DEFAULT_TICKERS + ", SHY, IEF, GLD"
        v6_capital = st.number_input("Capital simulado", min_value=1000, value=10000, step=1000)
        v6_universe = st.text_area("Universo (tickers)", value=default_v6_tickers, height=100)
        calc_v6 = st.button("Calcular cartera V6", type="primary", use_container_width=True)
        st.caption("Esto solo simula una cartera paper. No ejecuta operaciones.")
        st.caption("No es asesoramiento financiero.")
    else:
        st.markdown("### Configuración")
        tickers_text = st.text_area("Tickers", value=DEFAULT_TICKERS, height=90)
        op_type = st.radio("Tipo de operación", ["Swing 1-5 días", "Intradía experimental"])
        is_swing = op_type == "Swing 1-5 días"

        if is_swing:
            period_label = st.selectbox("Datos para probar", list(PERIOD_SWING.keys()), index=2)
            period_key = PERIOD_SWING[period_label]
            interval = "1d"
            holding = HOLDING_SWING[st.selectbox("Duración máxima de la operación", list(HOLDING_SWING.keys()), index=2)]
        else:
            period_label = st.selectbox("Datos para probar", list(PERIOD_INTRADAY.keys()), index=0)
            period_key = PERIOD_INTRADAY[period_label]
            interval = st.selectbox("Velas", ["15m", "30m", "1h"], index=1)
            holding = HOLDING_INTRADAY[st.selectbox("Duración máxima de la operación", list(HOLDING_INTRADAY.keys()), index=1)]

        st.caption("Ejemplo: 2 años = probamos con los últimos 2 años de datos. No significa que la operación dure 2 años.")
        if not is_swing:
            st.caption("⚠️ Intradía gratuito: experimental, no fiable para dinero real.")

        analyze = st.button("Analizar", type="primary", use_container_width=True)
        st.caption("No es asesoramiento financiero.")

if app_mode == "Portfolio V14 Approved Paper Trading":
    render_v14_mode(v14_capital, v14_universe, calc_v14)
elif app_mode == "Portfolio V6 Paper Trading":
    render_v6_mode(v6_capital, v6_universe, calc_v6)
else:
    st.markdown('<p class="title">📈 Trading Signals Lab</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Busca oportunidades de corto plazo con reglas testeadas. No es asesoramiento financiero.</p>', unsafe_allow_html=True)

    if not analyze:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2.5rem;">
            <div style="font-size:2.5rem;">📊</div>
            <p style="color:#9ca3af;">Escribe tickers en el panel izquierdo y pulsa <strong>Analizar</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        tickers = parse_tickers(tickers_text)
        if not tickers:
            st.error("Introduce al menos un ticker.")
        else:
            progress = st.progress(0, text="Iniciando...")
            mode_key = "swing" if is_swing else "intraday"
            results, errors = run_analysis(tickers, mode_key, period_key, interval, holding, progress)
            progress.empty()

            if not results:
                st.error("No se pudo analizar ningún ticker.")
            else:
                buy_list, hold_list, sell_list, avoid_list = categorize_results(results)
                main_mode, main_title, main_item = pick_main_feature(buy_list, hold_list, sell_list)
                analysis_map = build_analysis_map(results)

                # Tarjeta principal (nunca SELL como oportunidad)
                render_main_card(main_title, main_item, is_swing, holding, main_mode)
                render_action_card(main_item, main_mode)

                # Resumen rápido
                render_summary_counts(buy_list, hold_list, sell_list, avoid_list)

                # 4 secciones por tipo de señal
                render_signal_section(
                    "Oportunidades de compra",
                    "",
                    buy_list,
                    "No hay compras claras ahora.",
                )
                render_signal_section(
                    "En vigilancia",
                    "Activos fuertes o decentes, pero sin entrada clara ahora.",
                    hold_list,
                    "Ningún activo en vigilancia.",
                )
                render_signal_section(
                    "Alertas de salida",
                    "SELL significa salir/no comprar, no abrir cortos.",
                    sell_list,
                    "Sin alertas de salida.",
                )
                render_signal_section(
                    "Evitar",
                    "Demasiado riesgo, datos débiles o estrategia pobre.",
                    avoid_list,
                    "Ningún activo marcado como evitar.",
                )

                # Selector de gráfico
                ticker_options = [r.get("ticker") for r in results if r.get("ticker")]
                default_ticker = main_item.get("ticker") if main_item else (buy_list[0].get("ticker") if buy_list else ticker_options[0])
                if default_ticker not in ticker_options and ticker_options:
                    default_ticker = ticker_options[0]

                st.markdown('<div class="section-title">Gráfico</div>', unsafe_allow_html=True)
                selected_ticker = st.selectbox(
                    "Ver gráfico de",
                    options=ticker_options,
                    index=ticker_options.index(default_ticker) if default_ticker in ticker_options else 0,
                    label_visibility="collapsed",
                )
                selected_analysis = analysis_map.get(selected_ticker, {})
                st.markdown('<div class="card">', unsafe_allow_html=True)
                render_chart(selected_analysis, selected_ticker)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="section-title">Comparación con el mercado</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="section-hint">Estrategias para <strong>{selected_ticker}</strong></div>', unsafe_allow_html=True)
                render_market_comparison(selected_analysis.get("results_list", []))

                with st.expander("Ver explicación del algoritmo"):
                    st.markdown("""
                    Esta app no predice el futuro. Compara varias reglas simples:
                    - Tendencia por medias
                    - Ruptura de máximos
                    - Pullback en tendencia
                    - Rebote por sobreventa
                    - Debilidad para salir

                    Después compara cada regla contra comprar y mantener.
                    BUY solo aparece si hay entrada activa ahora y el backtest mínimo acompaña.
                    """)

                with st.expander("Ver datos técnicos"):
                    df = resolve_dataframe(selected_analysis.get("df"))
                    if df is not None:
                        last = df.iloc[-1]
                        cols = ["Close", "EMA_9", "EMA_21", "RSI_14", "MOMENTUM_5", "ATR_14"]
                        data = {c: round(float(last[c]), 2) for c in cols if c in last.index and pd.notna(last[c])}
                        st.json(data)
                    else:
                        st.text("No disponible")

                if errors:
                    with st.expander(f"Ver errores de tickers ({len(errors)})"):
                        for e in errors:
                            st.text(e)
