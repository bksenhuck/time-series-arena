"""Main dashboard — interactive forecast comparison: ARIMA vs XGBoost."""
import dash
from dash import Input, Output, State, callback, dcc, html

from frontend.callbacks.forecast_callbacks import (
    fetch_page_metrics,
    fetch_series,
    post_forecast,
    register_page_loader,
)
from frontend.components.charts import forecast_chart, historical_chart
from frontend.components.controls import (
    horizon_slider,
    model_radio,
    page_dropdown,
    run_button,
)
from frontend.components.layout import card, page_container, page_header
from frontend.config import THEME

dash.register_page(__name__, path="/", title="Dashboard")

T = THEME

# ── Layout ────────────────────────────────────────────────────────────────────

layout = page_container([
    page_header(
        "Forecast Dashboard",
        "Local ARIMA vs Global XGBoost — Wikipedia daily page views",
    ),

    # Controls
    card([
        html.Div([
            page_dropdown(),
            model_radio(),
            horizon_slider(),
            run_button(),
        ], style={
            "display": "flex", "gap": "24px",
            "flexWrap": "wrap", "alignItems": "flex-start",
        }),
    ], style={"marginBottom": "16px"}),

    # Status bar
    html.Div(id="dash-status", style={
        "color": T["text_muted"], "fontSize": "13px",
        "minHeight": "20px", "marginBottom": "8px",
    }),

    # Historical chart
    card([
        dcc.Loading(
            dcc.Graph(
                id="dash-historical",
                config={"displayModeBar": False},
                style={"height": "260px"},
            ),
            type="dot", color=T["arima_color"],
        ),
    ], style={"marginBottom": "16px"}),

    # Forecast chart
    card([
        dcc.Loading(
            dcc.Graph(
                id="dash-forecast",
                config={"displayModeBar": "hover"},
                style={"height": "360px"},
            ),
            type="dot", color=T["arima_color"],
        ),
    ], style={"marginBottom": "16px"}),

    # Metrics row
    html.Div([
        card([
            html.H4("ARIMA", style={
                "color": T["arima_color"], "margin": "0 0 12px",
                "fontSize": "14px", "fontWeight": "600",
            }),
            dcc.Loading(
                html.Div(id="dash-arima-metrics"),
                type="dot", color=T["arima_color"],
            ),
        ], style={"flex": "1"}),
        card([
            html.H4("XGBoost", style={
                "color": T["xgb_color"], "margin": "0 0 12px",
                "fontSize": "14px", "fontWeight": "600",
            }),
            dcc.Loading(
                html.Div(id="dash-xgb-metrics"),
                type="dot", color=T["xgb_color"],
            ),
        ], style={"flex": "1"}),
    ], style={"display": "flex", "gap": "16px"}),

    dcc.Store(id="dash-forecast-store"),
    dcc.Store(id="dash-pages-store"),
])

register_page_loader("page-selector", "dash-pages-store")


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("dash-historical", "figure"),
    Input("page-selector", "value"),
    prevent_initial_call=True,
)
def update_historical(page):
    if not page:
        return {}
    data = fetch_series(page)
    return historical_chart(data["dates"], data["views"], page) if data else {}


@callback(
    Output("dash-forecast-store", "data"),
    Output("dash-status", "children"),
    Input("run-btn", "n_clicks"),
    State("page-selector", "value"),
    State("model-selector", "value"),
    State("horizon-slider", "value"),
    prevent_initial_call=True,
)
def run_forecast(_, page, model, horizon):
    if not page:
        return {}, "⚠ Select a page first."
    return post_forecast(page, model, horizon)


@callback(
    Output("dash-forecast", "figure"),
    Input("dash-forecast-store", "data"),
    State("page-selector", "value"),
    prevent_initial_call=True,
)
def update_forecast_chart(store, page):
    if not store or not page:
        return {}
    data = fetch_series(page)
    if not data:
        return {}
    return forecast_chart(
        data["dates"], data["views"],
        arima_fc=store.get("arima"),
        xgb_fc=store.get("xgboost"),
        page=page,
    )


@callback(
    Output("dash-arima-metrics", "children"),
    Output("dash-xgb-metrics", "children"),
    Input("page-selector", "value"),
    prevent_initial_call=True,
)
def update_metrics(page):
    if not page:
        return _placeholder(), _placeholder()
    metrics = fetch_page_metrics(page)
    arima_m = next((m for m in metrics if m["model_type"] == "arima"), None)
    xgb_m   = next((m for m in metrics if m["model_type"] == "xgboost"), None)
    return _render_metric_card(arima_m), _render_metric_card(xgb_m)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _placeholder():
    return html.P("Select a page to see metrics.",
                  style={"color": T["text_muted"], "fontSize": "12px"})


def _render_metric_card(m):
    if not m:
        return html.P("Not yet computed — run the pipeline first.",
                      style={"color": T["text_muted"], "fontSize": "12px"})
    return html.Div([
        _metric_row(label, m.get(key))
        for key, label in [("mae", "MAE"), ("rmse", "RMSE"), ("mape", "MAPE %")]
    ])


def _metric_row(label: str, value):
    display = f"{value:.2f}" if value is not None else "—"
    return html.Div([
        html.Span(label, style={
            "color": T["text_muted"], "fontSize": "11px",
            "width": "62px", "display": "inline-block",
        }),
        html.Span(display, style={
            "color": T["text"], "fontWeight": "700", "fontSize": "17px",
        }),
    ], style={"marginBottom": "8px"})
