"""
Dataset Overview — aggregated evaluation metrics across all Wikipedia pages.
"""
import dash
import requests
from dash import Input, Output, callback, dcc, html

from frontend.components.charts import metrics_bar_chart, metrics_scatter_chart
from frontend.components.controls import metric_radio
from frontend.components.layout import card, page_container, page_header, stat_card
from frontend.config import API_BASE, THEME

dash.register_page(__name__, path="/dataset", title="Dataset Overview")

T = THEME
_API = API_BASE

_TH = {"color": T["text_muted"], "padding": "8px 12px", "textAlign": "left",
       "borderBottom": f"1px solid {T['border']}", "fontSize": "12px",
       "fontWeight": "600", "whiteSpace": "nowrap"}
_TD_BASE = {"padding": "6px 12px", "fontSize": "12px"}

layout = page_container([
    page_header(
        "Dataset Overview",
        "Walk-forward evaluation (14-day horizon) across all Wikipedia pages",
    ),

    # Summary stat cards
    dcc.Loading(
        html.Div(id="ds-summary-cards", style={
            "display": "flex", "gap": "16px",
            "marginBottom": "16px", "flexWrap": "wrap",
        }),
        type="dot", color=T["arima_color"],
    ),

    # Bar chart with metric toggle
    card([
        metric_radio("ds-metric-selector"),
        dcc.Loading(
            dcc.Graph(id="ds-bar-chart",
                      config={"displayModeBar": "hover"},
                      style={"height": "400px"}),
            type="dot", color=T["arima_color"],
        ),
    ], style={"marginBottom": "16px"}),

    # Scatter chart ARIMA MAE vs XGBoost MAE
    card([
        dcc.Loading(
            dcc.Graph(id="ds-scatter-chart",
                      config={"displayModeBar": "hover"},
                      style={"height": "420px"}),
            type="dot", color=T["xgb_color"],
        ),
    ], style={"marginBottom": "16px"}),

    # Detailed table
    card([
        html.H4("Detailed Results", style={
            "color": T["text"], "margin": "0 0 12px",
            "fontSize": "14px", "fontWeight": "600",
        }),
        dcc.Loading(
            html.Div(id="ds-table"),
            type="dot", color=T["text_muted"],
        ),
    ]),

    dcc.Store(id="ds-metrics-store"),
    dcc.Interval(id="ds-load-trigger", interval=500, max_intervals=1),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("ds-metrics-store", "data"),
    Input("ds-load-trigger", "n_intervals"),
)
def load_metrics(_):
    try:
        resp = requests.get(f"{_API}/api/metrics", timeout=15)
        return resp.json().get("metrics", [])
    except Exception:
        return []


@callback(
    Output("ds-summary-cards", "children"),
    Input("ds-metrics-store", "data"),
)
def render_summary(metrics):
    if not metrics:
        return html.P(
            "No metrics yet — run the training pipeline first.",
            style={"color": T["text_muted"], "fontSize": "13px"},
        )

    def _avg(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def _fmt(v, suffix=""):
        return f"{v:.1f}{suffix}" if v is not None else "—"

    arima = [m for m in metrics if m["model_type"] == "arima"]
    xgb   = [m for m in metrics if m["model_type"] == "xgboost"]

    winner_mae = "arima" if (_avg(arima, "mae") or 1e9) < (_avg(xgb, "mae") or 1e9) else "xgboost"

    cards = [
        stat_card("ARIMA (local)", T["arima_color"], [
            ("MAE",    _fmt(_avg(arima, "mae"))),
            ("RMSE",   _fmt(_avg(arima, "rmse"))),
            ("MAPE %", _fmt(_avg(arima, "mape"), "%")),
        ], f"{len(arima)} pages"),
        stat_card("XGBoost (global)", T["xgb_color"], [
            ("MAE",    _fmt(_avg(xgb, "mae"))),
            ("RMSE",   _fmt(_avg(xgb, "rmse"))),
            ("MAPE %", _fmt(_avg(xgb, "mape"), "%")),
        ], f"{len(xgb)} pages"),
        stat_card("Best MAE", T["text"], [
            ("Model",  "ARIMA" if winner_mae == "arima" else "XGBoost"),
            ("Pages",  str(len(arima) + len(xgb))),
        ], "walk-forward CV"),
    ]
    return cards


@callback(
    Output("ds-bar-chart", "figure"),
    Input("ds-metrics-store", "data"),
    Input("ds-metric-selector", "value"),
)
def render_bar(metrics, metric_key):
    return metrics_bar_chart(metrics, metric_key) if metrics else {}


@callback(
    Output("ds-scatter-chart", "figure"),
    Input("ds-metrics-store", "data"),
)
def render_scatter(metrics):
    return metrics_scatter_chart(metrics) if metrics else {}


@callback(
    Output("ds-table", "children"),
    Input("ds-metrics-store", "data"),
)
def render_table(metrics):
    if not metrics:
        return html.P("No data.", style={"color": T["text_muted"]})

    header = html.Tr([
        html.Th(col, style=_TH)
        for col in ["Page", "Model", "MAE", "RMSE", "MAPE %"]
    ])

    rows = []
    for m in sorted(metrics, key=lambda x: (x["series_id"], x["model_type"])):
        color = T["arima_color"] if m["model_type"] == "arima" else T["xgb_color"]
        rows.append(html.Tr([
            html.Td(m["series_id"].replace("_", " ")[:32],
                    style={**_TD_BASE, "color": T["text"]}),
            html.Td(m["model_type"].upper(),
                    style={**_TD_BASE, "color": color, "fontWeight": "600"}),
            html.Td(f"{m['mae']:.1f}"  if m.get("mae")  else "—",
                    style={**_TD_BASE, "color": T["text"]}),
            html.Td(f"{m['rmse']:.1f}" if m.get("rmse") else "—",
                    style={**_TD_BASE, "color": T["text"]}),
            html.Td(f"{m['mape']:.1f}%" if m.get("mape") else "—",
                    style={**_TD_BASE, "color": T["text"]}),
        ], style={"borderBottom": f"1px solid {T['border']}"}))

    return html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse"},
    )
