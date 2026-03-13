"""Reusable Plotly chart builders for the Time Series Arena dashboard."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from frontend.config import THEME

T = THEME


def _base_layout(title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(color=T["text"], size=16)),
        paper_bgcolor=T["surface"],
        plot_bgcolor=T["surface"],
        font=dict(color=T["text"], family="Inter, sans-serif"),
        xaxis=dict(
            gridcolor=T["border"],
            linecolor=T["border"],
            tickfont=dict(color=T["text_muted"]),
        ),
        yaxis=dict(
            gridcolor=T["border"],
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
    )


def historical_chart(dates, views, page: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=views,
        mode="lines",
        name="Daily Views",
        line=dict(color=T["history_color"], width=1.5),
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
) -> go.Figure:
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist_dates[-90:], y=hist_views[-90:],
        mode="lines",
        name="History (last 90d)",
        line=dict(color=T["history_color"], width=1.5),
    ))

    # ARIMA forecast
    if arima_fc:
        _add_forecast_band(fig, arima_fc, "ARIMA", T["arima_color"])

    # XGBoost forecast
    if xgb_fc:
        _add_forecast_band(fig, xgb_fc, "XGBoost", T["xgb_color"])

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
        line=dict(color=color, width=2, dash="dash"),
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
        line=dict(color=T["border"], dash="dot", width=1),
        showlegend=True,
    ))

    layout = _base_layout("MAE Scatter — ARIMA vs XGBoost (below diagonal = XGBoost wins)")
    layout["xaxis"]["title"] = "ARIMA MAE"
    layout["yaxis"]["title"] = "XGBoost MAE"
    fig.update_layout(**layout)
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
