"""
Home page — interactive forecast comparison: ARIMA vs XGBoost.
"""
import json

import dash
import requests
from dash import Input, Output, State, callback, dcc, html

from frontend.config import API_BASE, DEFAULT_HORIZON, MAX_HORIZON, MIN_HORIZON, THEME

dash.register_page(__name__, path="/legacy", title="Forecast (Legacy)")

T = THEME
_API = API_BASE


def _control_label(text: str):
    return html.Label(text, style={"color": T["text_muted"], "fontSize": "12px", "marginBottom": "4px"})


def _card(children, style: dict | None = None):
    base = {
        "backgroundColor": T["surface"],
        "border": f"1px solid {T['border']}",
        "borderRadius": "8px",
        "padding": "16px",
    }
    return html.Div(children, style={**base, **(style or {})})


# ── Layout ──────────────────────────────────────────────────────────────────

layout = html.Div([
    # Header
    html.Div([
        html.H2("Forecast Comparison", style={"color": T["text"], "margin": "0"}),
        html.P(
            "Compare local ARIMA vs global XGBoost forecasts on Wikipedia page views",
            style={"color": T["text_muted"], "margin": "4px 0 0"},
        ),
    ], style={"marginBottom": "20px"}),

    # Controls row
    _card([
        html.Div([
            # Page selector
            html.Div([
                _control_label("Wikipedia Page"),
                dcc.Dropdown(
                    id="page-selector",
                    placeholder="Select a page…",
                    style={"backgroundColor": T["surface2"], "color": T["text"]},
                    clearable=False,
                ),
            ], style={"flex": "2", "minWidth": "240px"}),

            # Model selector
            html.Div([
                _control_label("Model"),
                dcc.RadioItems(
                    id="model-selector",
                    options=[
                        {"label": " ARIMA", "value": "arima"},
                        {"label": " XGBoost", "value": "xgboost"},
                        {"label": " Both", "value": "both"},
                    ],
                    value="both",
                    labelStyle={"display": "block", "color": T["text"], "marginBottom": "4px"},
                    inputStyle={"marginRight": "6px"},
                ),
            ], style={"flex": "1", "minWidth": "140px"}),

            # Horizon slider
            html.Div([
                _control_label("Forecast Horizon (days)"),
                dcc.Slider(
                    id="horizon-slider",
                    min=MIN_HORIZON, max=MAX_HORIZON, step=1,
                    value=DEFAULT_HORIZON,
                    marks={MIN_HORIZON: str(MIN_HORIZON), MAX_HORIZON: str(MAX_HORIZON)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], style={"flex": "2", "minWidth": "240px"}),

            # Run button
            html.Div([
                html.Button(
                    "Run Forecast",
                    id="run-btn",
                    n_clicks=0,
                    style={
                        "backgroundColor": T["arima_color"],
                        "color": "#0f1117",
                        "border": "none",
                        "borderRadius": "6px",
                        "padding": "10px 24px",
                        "fontWeight": "600",
                        "cursor": "pointer",
                        "marginTop": "20px",
                    },
                ),
            ], style={"flex": "1", "minWidth": "120px", "display": "flex", "alignItems": "flex-end"}),
        ], style={"display": "flex", "gap": "24px", "flexWrap": "wrap", "alignItems": "flex-start"}),
    ], style={"marginBottom": "16px"}),

    # Status bar
    html.Div(id="status-bar", style={"color": T["text_muted"], "fontSize": "13px", "minHeight": "20px", "marginBottom": "8px"}),

    # Historical chart
    _card([
        dcc.Graph(id="historical-chart", config={"displayModeBar": False},
                  style={"height": "260px"}),
    ], style={"marginBottom": "16px"}),

    # Forecast chart
    _card([
        dcc.Graph(id="forecast-chart", config={"displayModeBar": "hover"},
                  style={"height": "360px"}),
    ], style={"marginBottom": "16px"}),

    # Metrics row
    html.Div([
        _card([
            html.H4("ARIMA", style={"color": T["arima_color"], "margin": "0 0 12px"}),
            html.Div(id="arima-metrics"),
        ], style={"flex": "1"}),
        _card([
            html.H4("XGBoost", style={"color": T["xgb_color"], "margin": "0 0 12px"}),
            html.Div(id="xgb-metrics"),
        ], style={"flex": "1"}),
    ], style={"display": "flex", "gap": "16px"}),

    # Store
    dcc.Store(id="forecast-store"),
    dcc.Store(id="pages-store"),
], style={"padding": "24px", "backgroundColor": T["bg"], "minHeight": "100vh"})


# ── Callbacks ────────────────────────────────────────────────────────────────

@callback(
    Output("page-selector", "options"),
    Output("page-selector", "value"),
    Output("pages-store", "data"),
    Input("page-selector", "id"),  # fires once on mount
)
def load_pages(_):
    try:
        resp = requests.get(f"{_API}/api/pages", timeout=10)
        pages = resp.json().get("pages", [])
    except Exception:
        pages = []
    options = [{"label": p.replace("_", " "), "value": p} for p in pages]
    default = pages[0] if pages else None
    return options, default, pages


@callback(
    Output("historical-chart", "figure"),
    Input("page-selector", "value"),
    prevent_initial_call=True,
)
def update_historical(page):
    if not page:
        return {}
    try:
        resp = requests.get(f"{_API}/api/series/{requests.utils.quote(page, safe='')}", timeout=10)
        data = resp.json()
        from frontend.components.charts import historical_chart
        return historical_chart(data["dates"], data["views"], page)
    except Exception:
        return {}


@callback(
    Output("forecast-store", "data"),
    Output("status-bar", "children"),
    Input("run-btn", "n_clicks"),
    State("page-selector", "value"),
    State("model-selector", "value"),
    State("horizon-slider", "value"),
    prevent_initial_call=True,
)
def run_forecast(_, page, model, horizon):
    if not page:
        return {}, "Select a page first."
    try:
        resp = requests.post(
            f"{_API}/api/forecast",
            json={"page": page, "model": model, "horizon": horizon},
            timeout=60,
        )
        if resp.status_code != 200:
            return {}, f"Error: {resp.json().get('detail', resp.text)}"
        return resp.json(), f"Forecast ready — {page.replace('_', ' ')} | {horizon}-day horizon"
    except Exception as exc:
        return {}, f"Request failed: {exc}"


@callback(
    Output("forecast-chart", "figure"),
    Input("forecast-store", "data"),
    State("page-selector", "value"),
    prevent_initial_call=True,
)
def update_forecast_chart(store, page):
    if not store or not page:
        return {}
    try:
        resp = requests.get(f"{_API}/api/series/{requests.utils.quote(page, safe='')}", timeout=10)
        data = resp.json()
    except Exception:
        return {}
    from frontend.components.charts import forecast_chart
    return forecast_chart(
        data["dates"], data["views"],
        arima_fc=store.get("arima"),
        xgb_fc=store.get("xgboost"),
        page=page,
    )


@callback(
    Output("arima-metrics", "children"),
    Output("xgb-metrics", "children"),
    Input("page-selector", "value"),
    prevent_initial_call=True,
)
def update_metrics_cards(page):
    if not page:
        return [], []
    try:
        resp = requests.get(f"{_API}/api/metrics/{requests.utils.quote(page, safe='')}", timeout=10)
        metrics = resp.json().get("metrics", [])
    except Exception:
        return [], []

    def _render(m):
        if not m:
            return html.P("Not computed yet", style={"color": T["text_muted"]})
        rows = []
        for key, label in [("mae", "MAE"), ("rmse", "RMSE"), ("mape", "MAPE %")]:
            val = m.get(key)
            rows.append(html.Div([
                html.Span(label, style={"color": T["text_muted"], "width": "70px", "display": "inline-block"}),
                html.Span(
                    f"{val:.2f}" if val is not None else "—",
                    style={"color": T["text"], "fontWeight": "600"},
                ),
            ], style={"marginBottom": "6px"}))
        return rows

    arima_m = next((m for m in metrics if m["model_type"] == "arima"), {})
    xgb_m = next((m for m in metrics if m["model_type"] == "xgboost"), {})
    return _render(arima_m), _render(xgb_m)
