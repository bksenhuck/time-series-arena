"""Reusable Plotly chart builders for the Time Series Arena dashboard."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    DEFAULT_LINE_WIDTH,
    FORECAST_TRAIN_HISTORY_DAYS,
    MAX_LINE_WIDTH,
    MIN_LINE_WIDTH,
    REAL_DASHED_LINE_WIDTH,
    THEME,
    TRAIN_TEST_SPLIT_DATE,
)

T = THEME

_MODEL_COLORS = {
    "arima":      T["arima_color"],
    "xgboost":    T["xgb_color"],
    "prophet":    T["prophet_color"],
    "auto_arima": T["auto_arima_color"],
}


def _base_layout(title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(color=T["text"], size=16)),
        paper_bgcolor=T["surface"],
        plot_bgcolor=T["surface"],
        font=dict(color=T["text"], family="Inter, sans-serif"),
        xaxis=dict(
            showgrid=False,
            linecolor=T["border"],
            tickfont=dict(color=T["text_muted"]),
        ),
        yaxis=dict(
            showgrid=False,
            linecolor=T["border"],
            tickfont=dict(color=T["text_muted"]),
        ),
        legend=dict(
            bgcolor=T["surface2"],
            bordercolor=T["border"],
            borderwidth=1,
            font=dict(color=T["text"]),
        ),
        margin=dict(l=50, r=20, t=50, b=40),
        hovermode="x unified",
        dragmode=False,
    )


def historical_chart(dates, views, page: str) -> go.Figure:
    fig = go.Figure()

    # Background split: train vs test period, controlled by config split date
    if dates:
        min_date = dates[0]
        max_date = dates[-1]
        split_date = TRAIN_TEST_SPLIT_DATE

        train_x1 = split_date if split_date < max_date else max_date
        if min_date <= train_x1:
            fig.add_vrect(
                x0=min_date,
                x1=train_x1,
                fillcolor=T.get("train_bg_color", "#2f4f3a"),
                opacity=T.get("train_test_bg_opacity", 0.42),
                layer="below",
                line_width=0,
            )

        if max_date > split_date:
            fig.add_vrect(
                x0=split_date,
                x1=max_date,
                fillcolor=T.get("test_bg_color", "#4f3a3a"),
                opacity=T.get("train_test_bg_opacity", 0.42),
                layer="below",
                line_width=0,
            )

    fig.add_trace(go.Scatter(
        x=dates, y=views,
        mode="lines",
        name="Daily Views",
        line=dict(color=T["history_color"], width=DEFAULT_LINE_WIDTH),
        fill="tozeroy",
        fillcolor=f"rgba(148,163,184,0.08)",
    ))
    fig.update_layout(**_base_layout(f"Historical Page Views — {page.replace('_', ' ')}"))
    return fig


def forecast_chart(
    hist_dates, hist_views,
    arima_fc: dict | None,
    xgb_fc: dict | None,
    page: str,
    prophet_fc: dict | None = None,
) -> go.Figure:
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist_dates[-90:], y=hist_views[-90:],
        mode="lines",
        name="History (last 90d)",
        line=dict(color=T["history_color"], width=DEFAULT_LINE_WIDTH),
    ))

    # ARIMA forecast
    if arima_fc:
        _add_forecast_band(fig, arima_fc, "ARIMA", T["arima_color"])

    # XGBoost forecast
    if xgb_fc:
        _add_forecast_band(fig, xgb_fc, "XGBoost", T["xgb_color"])

    # Prophet forecast
    if prophet_fc:
        _add_forecast_band(fig, prophet_fc, "Prophet", T["prophet_color"])

    fig.update_layout(**_base_layout(f"Forecast — {page.replace('_', ' ')}"))
    return fig


def _add_forecast_band(fig: go.Figure, fc: dict, name: str, color: str):
    rgba_fill = _hex_to_rgba(color, T["ci_opacity"])

    # CI upper (invisible, fills down)
    fig.add_trace(go.Scatter(
        x=fc["dates"], y=fc["upper"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    # CI lower (fills to upper)
    fig.add_trace(go.Scatter(
        x=fc["dates"], y=fc["lower"],
        mode="lines", line=dict(width=0),
        fill="tonexty",
        fillcolor=rgba_fill,
        showlegend=False, hoverinfo="skip",
    ))
    # Mean line
    fig.add_trace(go.Scatter(
        x=fc["dates"], y=fc["mean"],
        mode="lines+markers",
        name=name,
        line=dict(color=color, width=DEFAULT_LINE_WIDTH, dash="dash"),
        marker=dict(size=4),
    ))


def metrics_bar_chart(metrics: list[dict], metric_key: str = "mae") -> go.Figure:
    """Bar chart comparing ARIMA vs XGBoost on a given metric across pages."""
    arima_rows = [m for m in metrics if m["model_type"] == "arima" and m.get(metric_key) is not None]
    xgb_rows = [m for m in metrics if m["model_type"] == "xgboost" and m.get(metric_key) is not None]

    # Match pages present in both
    arima_map = {r["series_id"]: r[metric_key] for r in arima_rows}
    xgb_map = {r["series_id"]: r[metric_key] for r in xgb_rows}
    pages = sorted(set(arima_map) & set(xgb_map), key=lambda p: arima_map.get(p, 0))[:40]

    labels = [p.replace("_", " ")[:20] for p in pages]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=[arima_map[p] for p in pages],
        name="ARIMA", marker_color=T["arima_color"],
    ))
    fig.add_trace(go.Bar(
        x=labels, y=[xgb_map[p] for p in pages],
        name="XGBoost", marker_color=T["xgb_color"],
    ))

    layout = _base_layout(f"{metric_key.upper()} Comparison — ARIMA vs XGBoost (top 40 pages)")
    layout["barmode"] = "group"
    layout["xaxis"]["tickangle"] = -45
    layout["xaxis"]["tickfont"] = dict(size=9, color=T["text_muted"])
    fig.update_layout(**layout)
    return fig


def metrics_scatter_chart(metrics: list[dict]) -> go.Figure:
    """Scatter: ARIMA MAE vs XGBoost MAE per page."""
    arima_map = {m["series_id"]: m for m in metrics if m["model_type"] == "arima"}
    xgb_map = {m["series_id"]: m for m in metrics if m["model_type"] == "xgboost"}
    pages = list(set(arima_map) & set(xgb_map))

    x_vals = [arima_map[p]["mae"] for p in pages if arima_map[p]["mae"] is not None]
    y_vals = [xgb_map[p]["mae"] for p in pages if xgb_map[p]["mae"] is not None]
    labels = [p.replace("_", " ")[:20] for p in pages if arima_map[p]["mae"] is not None]

    max_val = max(max(x_vals or [1]), max(y_vals or [1]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont=dict(size=8, color=T["text_muted"]),
        marker=dict(size=8, color=T["xgb_color"], opacity=0.8),
        hovertemplate="<b>%{text}</b><br>ARIMA MAE: %{x:.1f}<br>XGBoost MAE: %{y:.1f}<extra></extra>",
    ))
    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", name="Equal performance",
        line=dict(color=T["border"], dash="dot", width=DEFAULT_LINE_WIDTH),
        showlegend=True,
    ))

    layout = _base_layout("MAE Scatter — ARIMA vs XGBoost (below diagonal = XGBoost wins)")
    layout["xaxis"]["title"] = "ARIMA MAE"
    layout["yaxis"]["title"] = "XGBoost MAE"
    fig.update_layout(**layout)
    return fig


def empty_figure(message: str = "Select a page to begin") -> go.Figure:
    """Dark-styled empty/placeholder figure for unloaded chart slots."""
    fig = go.Figure()
    fig.update_layout(**_base_layout())
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(color=T["text_muted"], size=13),
        bgcolor="rgba(0,0,0,0)",
    )
    return fig


def multi_horizon_forecast_chart(
    hist_dates,
    hist_views,
    all_forecasts: dict,
    page: str,
) -> go.Figure:
    """
    Plot history + all pre-computed horizon forecasts per model.

    all_forecasts: {model_name: {horizon_int: forecast_dict, ...}, ...}
    Each model gets its theme colour; horizons get increasing opacity (7d=faint → 50d=solid).
    """
    fig = go.Figure()

    # ── Build opacity / width maps ────────────────────────────
    # JSON round-trip turns int keys into strings — normalise to int
    all_forecasts = {
        m: {int(h): fc for h, fc in by_h.items()}
        for m, by_h in all_forecasts.items()
    }
    all_horizons = sorted({h for fc in all_forecasts.values() for h in fc})
    n = max(len(all_horizons) - 1, 1)
    opacity_map = {h: 0.45 + 0.55 * i / n for i, h in enumerate(all_horizons)}

    # ── Split hist into training part and out-of-sample "real values" ──
    # The first forecast date is the day after the training cutoff,
    # so any stored hist data on or after that date is genuinely out-of-sample.
    first_fc_date: str | None = None
    for fc_by_h in all_forecasts.values():
        for fc in fc_by_h.values():
            if fc.get("dates"):
                candidate = fc["dates"][0]
                if first_fc_date is None or candidate < first_fc_date:
                    first_fc_date = candidate

    if first_fc_date:
        split = next(
            (i for i, d in enumerate(hist_dates) if d >= first_fc_date),
            len(hist_dates),
        )
        train_dates = hist_dates[:split]
        train_views = hist_views[:split]
        real_dates  = hist_dates[split:]
        real_views  = hist_views[split:]
    else:
        train_dates = hist_dates
        train_views = hist_views
        real_dates  = []
        real_views  = []

    # ── Historical line (configurable training window) ───────────────
    if FORECAST_TRAIN_HISTORY_DAYS > 0:
        hist_window_dates = train_dates[-FORECAST_TRAIN_HISTORY_DAYS:]
        hist_window_views = train_views[-FORECAST_TRAIN_HISTORY_DAYS:]
        hist_name = f"Histórico treino ({FORECAST_TRAIN_HISTORY_DAYS}d)"
    else:
        hist_window_dates = train_dates
        hist_window_views = train_views
        hist_name = "Histórico treino (completo)"

    fig.add_trace(go.Scatter(
        x=hist_window_dates, y=hist_window_views,
        mode="lines",
        name=hist_name,
        showlegend=False,
        line=dict(color=T["history_color"], width=DEFAULT_LINE_WIDTH),
    ))

    # ── Out-of-sample "real values" overlay ──────────────────────────
    if real_dates:
        fig.add_trace(go.Scatter(
            x=real_dates, y=real_views,
            mode="lines",
            name="Valor real",
            showlegend=False,
            line=dict(color=T["history_color"], width=REAL_DASHED_LINE_WIDTH, dash="dot"),
            hovertemplate="<b>Real</b><br>%{x}<br>%{y:,.0f} views<extra></extra>",
        ))

    # ── Forecast lines per model ──────────────────────────────
    for model_name, fc_by_horizon in all_forecasts.items():
        color = _MODEL_COLORS.get(model_name, T["text_muted"])
        r, g, b = _hex_to_rgb(color)

        for h in sorted(fc_by_horizon):
            fc = fc_by_horizon[h]
            opacity = opacity_map.get(h, 1.0)
            line_color = f"rgba({r},{g},{b},{opacity})"

            fig.add_trace(go.Scatter(
                x=fc["dates"], y=fc["mean"],
                mode="lines",
                name=f"{model_name.upper()} {h}d",
                legendgroup=model_name,
                showlegend=False,
                line=dict(color=line_color, width=DEFAULT_LINE_WIDTH),
                hovertemplate=(
                    f"<b>{model_name.upper()} — {h}d</b><br>"
                    "%{x}<br>%{y:,.0f} views<extra></extra>"
                ),
            ))

    layout = _base_layout(f"Forecast — {page.replace('_', ' ')}")
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def _split_hist_by_forecast_start(hist_dates, hist_views, all_forecasts: dict):
    """Split history into train/test parts based on the first forecast date."""
    first_fc_date: str | None = None
    for fc_by_h in all_forecasts.values():
        for fc in fc_by_h.values():
            if fc.get("dates"):
                candidate = fc["dates"][0][:10]  # normalise to YYYY-MM-DD
                if first_fc_date is None or candidate < first_fc_date:
                    first_fc_date = candidate

    if not first_fc_date:
        return hist_dates, hist_views, [], []

    split = next(
        (i for i, d in enumerate(hist_dates) if d[:10] >= first_fc_date),
        len(hist_dates),
    )
    return hist_dates[:split], hist_views[:split], hist_dates[split:], hist_views[split:]


def _add_model_lines(fig: go.Figure, all_forecasts: dict, fixed_opacity: float | None = None):
    """Add multi-horizon model lines with consistent opacity/width mapping.

    fixed_opacity: when set, all lines use that opacity (ignores horizon-based scaling).
    """
    all_forecasts = {
        m: {int(h): fc for h, fc in by_h.items()}
        for m, by_h in all_forecasts.items()
    }
    all_horizons = sorted({h for fc in all_forecasts.values() for h in fc})
    n = max(len(all_horizons) - 1, 1)
    opacity_map = {h: 0.45 + 0.55 * i / n for i, h in enumerate(all_horizons)}

    for model_name, fc_by_horizon in all_forecasts.items():
        color = _MODEL_COLORS.get(model_name, T["text_muted"])
        r, g, b = _hex_to_rgb(color)

        for h in sorted(fc_by_horizon):
            fc = fc_by_horizon[h]
            opacity = fixed_opacity if fixed_opacity is not None else opacity_map.get(h, 1.0)
            line_color = f"rgba({r},{g},{b},{opacity})"

            fig.add_trace(go.Scatter(
                x=fc["dates"], y=fc["mean"],
                mode="lines",
                name=f"{model_name.upper()} {h}d",
                legendgroup=model_name,
                showlegend=False,
                line=dict(color=line_color, width=DEFAULT_LINE_WIDTH),
                hovertemplate=(
                    f"<b>{model_name.upper()} — {h}d</b><br>"
                    "%{x}<br>%{y:,.0f} views<extra></extra>"
                ),
            ))


def multi_horizon_train_chart(
    hist_dates,
    hist_views,
    all_forecasts: dict,
    page: str,
) -> go.Figure:
    """Forecast-style chart focused on training data + model lines (horizon-based)."""
    fig = go.Figure()

    train_dates, train_views, _, _ = _split_hist_by_forecast_start(
        hist_dates, hist_views, all_forecasts
    )
    if FORECAST_TRAIN_HISTORY_DAYS > 0:
        train_dates = train_dates[-FORECAST_TRAIN_HISTORY_DAYS:]
        train_views = train_views[-FORECAST_TRAIN_HISTORY_DAYS:]

    fig.add_trace(go.Scatter(
        x=train_dates, y=train_views,
        mode="lines",
        name="Histórico treino",
        showlegend=False,
        line=dict(color=T["history_color"], width=DEFAULT_LINE_WIDTH),
    ))

    _add_model_lines(fig, all_forecasts)

    layout = _base_layout(f"Forecast (Treino) — {page.replace('_', ' ')}")
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def train_fitted_models_chart(train_fit: dict, page: str) -> go.Figure:
    """Train chart with dotted real line + in-sample fitted lines per model."""
    fig = go.Figure()

    actual = train_fit.get("actual") or {}
    if actual.get("dates") and actual.get("mean"):
        fig.add_trace(go.Scatter(
            x=actual["dates"], y=actual["mean"],
            mode="lines",
            name="Valor real",
            showlegend=False,
            line=dict(color=T["history_color"], width=REAL_DASHED_LINE_WIDTH, dash="dot"),
            hovertemplate="<b>Real</b><br>%{x}<br>%{y:,.0f} views<extra></extra>",
        ))

    for model in ("arima", "xgboost", "prophet", "auto_arima"):
        payload = train_fit.get(model) or {}
        if not payload.get("dates") or not payload.get("mean"):
            continue
        fig.add_trace(go.Scatter(
            x=payload["dates"], y=payload["mean"],
            mode="lines",
            name=model.upper(),
            showlegend=False,
            line=dict(color=_MODEL_COLORS.get(model, T["text_muted"]), width=DEFAULT_LINE_WIDTH),
            hovertemplate=(
                f"<b>{model.upper()} — treino</b><br>"
                "%{x}<br>%{y:,.0f} views<extra></extra>"
            ),
        ))

    layout = _base_layout(f"Forecast (Treino) — {page.replace('_', ' ')}")
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def multi_horizon_test_chart(
    hist_dates,
    hist_views,
    all_forecasts: dict,
    page: str,
) -> go.Figure:
    """Forecast-style chart focused on test/real data + model lines."""
    fig = go.Figure()

    _, _, test_dates, test_views = _split_hist_by_forecast_start(
        hist_dates, hist_views, all_forecasts
    )

    if test_dates:
        fig.add_trace(go.Scatter(
            x=test_dates, y=test_views,
            mode="lines",
            name="Valor real",
            showlegend=False,
            line=dict(color=T["history_color"], width=REAL_DASHED_LINE_WIDTH, dash="dot"),
            hovertemplate="<b>Real</b><br>%{x}<br>%{y:,.0f} views<extra></extra>",
        ))

    _add_model_lines(fig, all_forecasts, fixed_opacity=1.0)

    layout = _base_layout(f"Forecast (Teste) — {page.replace('_', ' ')}")
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def training_trend_chart(hist_dates, hist_views, page: str) -> go.Figure:
    """Trend chart for training-only data (up to TRAIN_TEST_SPLIT_DATE)."""
    fig = go.Figure()

    split = next(
        (i for i, d in enumerate(hist_dates) if d > TRAIN_TEST_SPLIT_DATE),
        len(hist_dates),
    )
    train_dates = hist_dates[:split]
    train_views = hist_views[:split]

    if FORECAST_TRAIN_HISTORY_DAYS > 0:
        train_dates = train_dates[-FORECAST_TRAIN_HISTORY_DAYS:]
        train_views = train_views[-FORECAST_TRAIN_HISTORY_DAYS:]

    fig.add_trace(go.Scatter(
        x=train_dates,
        y=train_views,
        mode="lines",
        name="Treino",
        showlegend=False,
        line=dict(color=T["history_color"], width=DEFAULT_LINE_WIDTH),
        fill="tozeroy",
        fillcolor=f"rgba(148,163,184,0.08)",
        hovertemplate="<b>Treino</b><br>%{x}<br>%{y:,.0f} views<extra></extra>",
    ))

    layout = _base_layout(f"Training Trend — {page.replace('_', ' ')}")
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"
