"""
Metrics page — global model performance comparison across all pages.
"""
import dash
import requests
from dash import Input, Output, callback, dcc, html

from frontend.config import API_BASE, THEME

dash.register_page(__name__, path="/metrics", title="Model Metrics")

T = THEME
_API = API_BASE


def _card(children, style: dict | None = None):
    base = {
        "backgroundColor": T["surface"],
        "border": f"1px solid {T['border']}",
        "borderRadius": "8px",
        "padding": "16px",
    }
    return html.Div(children, style={**base, **(style or {})})


layout = html.Div([
    html.Div([
        html.H2("Model Performance", style={"color": T["text"], "margin": "0"}),
        html.P(
            "Evaluation metrics across all Wikipedia pages (walk-forward validation, 14-day horizon)",
            style={"color": T["text_muted"], "margin": "4px 0 0"},
        ),
    ], style={"marginBottom": "20px"}),

    # Summary cards row
    html.Div(id="summary-cards", style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"}),

    # Metric selector + bar chart
    _card([
        html.Div([
            html.Label("Metric:", style={"color": T["text_muted"], "marginRight": "12px"}),
            dcc.RadioItems(
                id="metric-selector",
                options=[
                    {"label": " MAE", "value": "mae"},
                    {"label": " RMSE", "value": "rmse"},
                    {"label": " MAPE %", "value": "mape"},
                ],
                value="mae",
                labelStyle={"display": "inline-block", "color": T["text"], "marginRight": "20px"},
                inputStyle={"marginRight": "6px"},
            ),
        ], style={"marginBottom": "16px"}),
        dcc.Graph(id="metrics-bar-chart", config={"displayModeBar": "hover"}, style={"height": "400px"}),
    ], style={"marginBottom": "16px"}),

    # Scatter chart
    _card([
        dcc.Graph(id="metrics-scatter-chart", config={"displayModeBar": "hover"}, style={"height": "420px"}),
    ], style={"marginBottom": "16px"}),

    # Data table
    _card([
        html.H4("Detailed Results", style={"color": T["text"], "margin": "0 0 12px"}),
        html.Div(id="metrics-table"),
    ]),

    dcc.Store(id="all-metrics-store"),
    dcc.Interval(id="metrics-load-trigger", interval=500, max_intervals=1),
], style={"padding": "24px", "backgroundColor": T["bg"], "minHeight": "100vh"})


# ── Callbacks ────────────────────────────────────────────────────────────────

@callback(
    Output("all-metrics-store", "data"),
    Input("metrics-load-trigger", "n_intervals"),
)
def load_metrics(_):
    try:
        resp = requests.get(f"{_API}/api/metrics", timeout=15)
        return resp.json().get("metrics", [])
    except Exception:
        return []


@callback(
    Output("summary-cards", "children"),
    Input("all-metrics-store", "data"),
)
def render_summary_cards(metrics):
    if not metrics:
        return html.P("No metrics computed yet. Run the training pipeline first.",
                      style={"color": T["text_muted"]})

    def avg(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    arima = [m for m in metrics if m["model_type"] == "arima"]
    xgb = [m for m in metrics if m["model_type"] == "xgboost"]

    def _stat_card(title, color, rows):
        mae_v = avg(rows, "mae")
        rmse_v = avg(rows, "rmse")
        mape_v = avg(rows, "mape")
        return _card([
            html.H4(title, style={"color": color, "margin": "0 0 12px"}),
            html.Div([
                html.Div([
                    html.Div("MAE", style={"color": T["text_muted"], "fontSize": "11px"}),
                    html.Div(f"{mae_v:.0f}" if mae_v else "—", style={"color": T["text"], "fontSize": "22px", "fontWeight": "700"}),
                ], style={"marginBottom": "8px"}),
                html.Div([
                    html.Div("RMSE", style={"color": T["text_muted"], "fontSize": "11px"}),
                    html.Div(f"{rmse_v:.0f}" if rmse_v else "—", style={"color": T["text"], "fontSize": "22px", "fontWeight": "700"}),
                ], style={"marginBottom": "8px"}),
                html.Div([
                    html.Div("MAPE %", style={"color": T["text_muted"], "fontSize": "11px"}),
                    html.Div(f"{mape_v:.1f}%" if mape_v else "—", style={"color": T["text"], "fontSize": "22px", "fontWeight": "700"}),
                ]),
            ]),
            html.Div(f"{len(rows)} pages", style={"color": T["text_muted"], "fontSize": "11px", "marginTop": "12px"}),
        ], style={"flex": "1", "minWidth": "180px"})

    return [
        _stat_card("ARIMA (local)", T["arima_color"], arima),
        _stat_card("XGBoost (global)", T["xgb_color"], xgb),
    ]


@callback(
    Output("metrics-bar-chart", "figure"),
    Input("all-metrics-store", "data"),
    Input("metric-selector", "value"),
)
def render_bar_chart(metrics, metric_key):
    if not metrics:
        return {}
    from frontend.components.charts import metrics_bar_chart
    return metrics_bar_chart(metrics, metric_key)


@callback(
    Output("metrics-scatter-chart", "figure"),
    Input("all-metrics-store", "data"),
)
def render_scatter(metrics):
    if not metrics:
        return {}
    from frontend.components.charts import metrics_scatter_chart
    return metrics_scatter_chart(metrics)


@callback(
    Output("metrics-table", "children"),
    Input("all-metrics-store", "data"),
)
def render_table(metrics):
    if not metrics:
        return html.P("No data", style={"color": T["text_muted"]})

    header = html.Tr([
        html.Th(col, style={"color": T["text_muted"], "padding": "8px 12px", "textAlign": "left",
                            "borderBottom": f"1px solid {T['border']}", "fontSize": "12px"})
        for col in ["Page", "Model", "MAE", "RMSE", "MAPE %"]
    ])

    rows = []
    for m in sorted(metrics, key=lambda x: (x["series_id"], x["model_type"])):
        color = T["arima_color"] if m["model_type"] == "arima" else T["xgb_color"]
        rows.append(html.Tr([
            html.Td(m["series_id"].replace("_", " ")[:30], style={"padding": "6px 12px", "color": T["text"], "fontSize": "12px"}),
            html.Td(m["model_type"].upper(), style={"padding": "6px 12px", "color": color, "fontSize": "12px", "fontWeight": "600"}),
            html.Td(f"{m['mae']:.1f}" if m["mae"] else "—", style={"padding": "6px 12px", "color": T["text"], "fontSize": "12px"}),
            html.Td(f"{m['rmse']:.1f}" if m["rmse"] else "—", style={"padding": "6px 12px", "color": T["text"], "fontSize": "12px"}),
            html.Td(f"{m['mape']:.1f}%" if m["mape"] else "—", style={"padding": "6px 12px", "color": T["text"], "fontSize": "12px"}),
        ], style={"borderBottom": f"1px solid {T['border']}"}))

    return html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse"},
    )
