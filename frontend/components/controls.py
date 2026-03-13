"""Reusable control components (dropdowns, sliders, radio buttons)."""
from dash import dcc, html

from frontend.components.layout import control_label
from frontend.config import DEFAULT_HORIZON, MAX_HORIZON, MIN_HORIZON, THEME

T = THEME

_DROPDOWN_STYLE = {
    "backgroundColor": T["surface2"],
    "color": T["text"],
    "border": f"1px solid {T['border']}",
    "borderRadius": "6px",
}

_BTN_PRIMARY = {
    "backgroundColor": T["arima_color"],
    "color": "#0f1117",
    "border": "none",
    "borderRadius": "6px",
    "padding": "10px 24px",
    "fontWeight": "600",
    "cursor": "pointer",
    "marginTop": "20px",
    "transition": "opacity .15s",
}


def page_dropdown(component_id: str = "page-selector") -> html.Div:
    return html.Div([
        control_label("Wikipedia Page"),
        dcc.Dropdown(
            id=component_id,
            placeholder="Select a page…",
            clearable=False,
            style=_DROPDOWN_STYLE,
        ),
    ], style={"flex": "2", "minWidth": "240px"})


def model_radio(component_id: str = "model-selector") -> html.Div:
    return html.Div([
        control_label("Model"),
        dcc.RadioItems(
            id=component_id,
            options=[
                {"label": "  ARIMA (local)",    "value": "arima"},
                {"label": "  XGBoost (global)", "value": "xgboost"},
                {"label": "  Both",             "value": "both"},
            ],
            value="both",
            labelStyle={"display": "block", "color": T["text"], "marginBottom": "6px"},
            inputStyle={"marginRight": "8px", "accentColor": T["arima_color"]},
        ),
    ], style={"flex": "1", "minWidth": "160px"})


def horizon_slider(component_id: str = "horizon-slider") -> html.Div:
    marks = {
        MIN_HORIZON: {"label": str(MIN_HORIZON), "style": {"color": T["text_muted"]}},
        MAX_HORIZON: {"label": str(MAX_HORIZON), "style": {"color": T["text_muted"]}},
    }
    return html.Div([
        control_label("Forecast Horizon (days)"),
        dcc.Slider(
            id=component_id,
            min=MIN_HORIZON,
            max=MAX_HORIZON,
            step=1,
            value=DEFAULT_HORIZON,
            marks=marks,
            tooltip={"placement": "bottom", "always_visible": True},
        ),
    ], style={"flex": "2", "minWidth": "240px"})


def run_button(component_id: str = "run-btn", label: str = "Run Forecast") -> html.Div:
    return html.Div([
        html.Button(label, id=component_id, n_clicks=0, style=_BTN_PRIMARY),
    ], style={"flex": "1", "minWidth": "120px",
               "display": "flex", "alignItems": "flex-end"})


def metric_radio(component_id: str = "metric-selector") -> html.Div:
    return html.Div([
        html.Span("Metric:", style={"color": T["text_muted"],
                                    "fontSize": "12px", "marginRight": "12px"}),
        dcc.RadioItems(
            id=component_id,
            options=[
                {"label": "  MAE",    "value": "mae"},
                {"label": "  RMSE",   "value": "rmse"},
                {"label": "  MAPE %", "value": "mape"},
            ],
            value="mae",
            inline=True,
            labelStyle={"color": T["text"], "marginRight": "20px"},
            inputStyle={"marginRight": "6px", "accentColor": T["arima_color"]},
        ),
    ], style={"marginBottom": "16px", "display": "flex",
              "alignItems": "center", "flexWrap": "wrap"})
