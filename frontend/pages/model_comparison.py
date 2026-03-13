"""
Model Comparison — side-by-side ARIMA vs XGBoost across up to 6 pages.
"""
import dash
import requests
from dash import Input, Output, State, callback, dcc, html

from frontend.callbacks.forecast_callbacks import fetch_series
from frontend.components.charts import forecast_chart
from frontend.components.controls import horizon_slider
from frontend.components.layout import card, control_label, page_container, page_header
from frontend.config import API_BASE, THEME

dash.register_page(__name__, path="/compare", title="Model Comparison")

T = THEME
_API = API_BASE

_BTN = {
    "backgroundColor": T["xgb_color"],
    "color": "#0f1117",
    "border": "none",
    "borderRadius": "6px",
    "padding": "10px 24px",
    "fontWeight": "600",
    "cursor": "pointer",
    "marginTop": "20px",
    "transition": "opacity .15s",
}

layout = page_container([
    page_header(
        "Model Comparison",
        "ARIMA (local) vs XGBoost (global) — up to 6 Wikipedia pages at once",
    ),

    card([
        html.Div([
            html.Div([
                control_label("Pages (up to 6)"),
                dcc.Dropdown(
                    id="cmp-pages",
                    multi=True,
                    placeholder="Select pages…",
                    maxHeight=320,
                    style={
                        "backgroundColor": T["surface2"],
                        "color": T["text"],
                        "minWidth": "320px",
                    },
                ),
            ], style={"flex": "3", "minWidth": "300px"}),

            horizon_slider("cmp-horizon"),

            html.Div([
                html.Button("Compare", id="cmp-btn", n_clicks=0, style=_BTN),
            ], style={"flex": "1", "minWidth": "120px",
                      "display": "flex", "alignItems": "flex-end"}),
        ], style={
            "display": "flex", "gap": "24px",
            "flexWrap": "wrap", "alignItems": "flex-start",
        }),
    ], style={"marginBottom": "16px"}),

    html.Div(id="cmp-status", style={
        "color": T["text_muted"], "fontSize": "13px",
        "minHeight": "20px", "marginBottom": "8px",
    }),

    dcc.Loading(
        html.Div(id="cmp-charts-grid"),
        type="dot", color=T["xgb_color"],
    ),

    dcc.Store(id="cmp-pages-store"),
    dcc.Interval(id="cmp-load-trigger", interval=400, max_intervals=1),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("cmp-pages", "options"),
    Output("cmp-pages", "value"),
    Output("cmp-pages-store", "data"),
    Input("cmp-load-trigger", "n_intervals"),
)
def load_pages(_):
    try:
        resp = requests.get(f"{_API}/api/pages", timeout=10)
        pages = resp.json().get("pages", [])
    except Exception:
        pages = []
    options = [{"label": p.replace("_", " "), "value": p} for p in pages]
    defaults = pages[:4] if len(pages) >= 4 else pages
    return options, defaults, pages


@callback(
    Output("cmp-charts-grid", "children"),
    Output("cmp-status", "children"),
    Input("cmp-btn", "n_clicks"),
    State("cmp-pages", "value"),
    State("cmp-horizon", "value"),
    prevent_initial_call=True,
)
def compare_pages(_, pages, horizon):
    if not pages:
        return [], "⚠ Select at least one page."

    pages = pages[:6]
    charts = []
    errors = []

    for page in pages:
        try:
            resp = requests.post(
                f"{_API}/api/forecast",
                json={"page": page, "model": "both", "horizon": horizon},
                timeout=60,
            )
            fc = resp.json() if resp.status_code == 200 else {}
            data = fetch_series(page)
            if not data:
                errors.append(page)
                continue
            fig = forecast_chart(
                data["dates"], data["views"],
                arima_fc=fc.get("arima"),
                xgb_fc=fc.get("xgboost"),
                page=page,
            )
            charts.append(
                card([
                    dcc.Graph(
                        figure=fig,
                        config={"displayModeBar": "hover"},
                        style={"height": "300px"},
                    ),
                ], style={"flex": "1", "minWidth": "46%"})
            )
        except Exception as exc:
            errors.append(f"{page} ({exc})")
            continue

    grid = html.Div(
        charts,
        style={"display": "flex", "flexWrap": "wrap", "gap": "16px"},
    )
    status = f"✓ {len(charts)} pages compared | {horizon}-day horizon"
    if errors:
        status += f" | ⚠ failed: {', '.join(errors)}"
    return grid, status
