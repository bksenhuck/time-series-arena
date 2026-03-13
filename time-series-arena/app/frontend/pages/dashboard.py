"""Main dashboard — interactive forecast comparison: ARIMA vs XGBoost."""
import dash
import plotly.graph_objects as go
import requests
from dash import Input, Output, callback, dcc, html

from app.frontend.callbacks.forecast_callbacks import (
    fetch_all_horizon_forecasts,
    fetch_page_metrics,
    fetch_scenario_comparison,
    fetch_series,
    fetch_train_fit_forecasts,
)
from app.frontend.components.charts import (
    empty_figure,
    historical_chart,
    train_fitted_models_chart,
    multi_horizon_test_chart,
    multi_horizon_train_chart,
)
from app.frontend.components.controls import (
    page_dropdown,
)
from app.frontend.components.layout import card, page_container, page_header
from config import API_BASE, THEME

dash.register_page(__name__, path="/", title="Dashboard")

T = THEME

# ── Layout ────────────────────────────────────────────────────────────────────


def layout() -> html.Div:
    """Called on every page visit — pre-populates the page dropdown via backend API."""
    try:
        resp = requests.get(f"{API_BASE}/api/pages", timeout=10)
        resp.raise_for_status()
        pages = resp.json().get("pages", [])
    except Exception:
        pages = []
    options = [{"label": p.replace("_", " "), "value": p} for p in pages]
    default = pages[0] if pages else None

    return page_container([
        page_header(
            "Forecast Dashboard",
            "Local ARIMA vs Global XGBoost — Wikipedia daily page views",
        ),

        # Controls
        card([
            html.Div([
                page_dropdown(options=options, value=default),
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
            html.Div([
                html.Span("Treino", style={
                    "backgroundColor": T["train_bg_color"],
                    "color": T["text"],
                    "opacity": 1,
                    "padding": "4px 10px",
                    "borderRadius": "6px",
                    "display": "inline-block",
                    "fontWeight": "600",
                    "fontSize": "13px",
                    "marginRight": "24px",
                }),
                html.Span("Teste", style={
                    "backgroundColor": T["test_bg_color"],
                    "color": T["text"],
                    "opacity": 1,
                    "padding": "4px 10px",
                    "borderRadius": "6px",
                    "display": "inline-block",
                    "fontWeight": "600",
                    "fontSize": "13px",
                }),
            ], style={"textAlign": "center", "marginBottom": "8px"}),
            dcc.Loading(
                dcc.Graph(
                    id="dash-historical",
                    figure=empty_figure("Selecione uma página para ver o histórico"),
                    config={"displayModeBar": False, "scrollZoom": False},
                    style={"height": "260px"},
                ),
                type="dot", color=T["arima_color"],
            ),
        ], style={"marginBottom": "16px"}),

        # Forecast chart (Treino)
        card([
            # ── Model colour legend ──────────────────────────────────────
            html.Div([
                html.Span("— ARIMA", style={
                    "color": T["arima_color"], "fontWeight": "600",
                    "fontSize": "13px", "marginRight": "24px",
                }),
                html.Span("— XGBoost", style={
                    "color": T["xgb_color"], "fontWeight": "600",
                    "fontSize": "13px", "marginRight": "24px",
                }),
                html.Span("— Prophet", style={
                    "color": T["prophet_color"], "fontWeight": "600",
                    "fontSize": "13px", "marginRight": "24px",
                }),                html.Span("— AutoARIMA", style={
                    "color": T["auto_arima_color"], "fontWeight": "600",
                    "fontSize": "13px", "marginRight": "24px",
                }),                html.Span("┈┈ Valor real", style={
                    "color": T["history_color"], "fontWeight": "400",
                    "fontSize": "13px",
                }),
            ], style={"textAlign": "center", "marginBottom": "8px"}),
            # ── Horizon explanation ───────────────────────────────────
            html.P(
                "Cada modelo é plotado em múltiplos horizontes (7, 14, 20, 25, 30 e 50 dias). "
                "Linhas mais grossas indicam horizontes mais curtos (previsões mais precisas); "
                "linhas mais finas e transparentes indicam horizontes mais longos.",
                style={
                    "color": T["text_muted"], "fontSize": "12px",
                    "textAlign": "center", "margin": "0 0 10px",
                    "lineHeight": "1.6",
                },
            ),
            dcc.Loading(
                dcc.Graph(
                    id="dash-forecast-train",
                    figure=empty_figure("Escolha uma página para ver o Forecast de Treino"),
                    config={"displayModeBar": False, "scrollZoom": False},
                    style={"height": "520px"},
                ),
                type="dot", color=T["arima_color"],
            ),
        ], style={"marginBottom": "16px"}),

        # Forecast chart (Teste)
        card([
            html.Div([
                html.Span("— ARIMA", style={
                    "color": T["arima_color"], "fontWeight": "600",
                    "fontSize": "13px", "marginRight": "24px",
                }),
                html.Span("— XGBoost", style={
                    "color": T["xgb_color"], "fontWeight": "600",
                    "fontSize": "13px", "marginRight": "24px",
                }),
                html.Span("— Prophet", style={
                    "color": T["prophet_color"], "fontWeight": "600",
                    "fontSize": "13px", "marginRight": "24px",
                }),
                html.Span("— AutoARIMA", style={
                    "color": T["auto_arima_color"], "fontWeight": "600",
                    "fontSize": "13px", "marginRight": "24px",
                }),
                html.Span("┈┈ Valor real", style={
                    "color": T["history_color"], "fontWeight": "400",
                    "fontSize": "13px",
                }),
            ], style={"textAlign": "center", "marginBottom": "8px"}),
            html.P(
                "Mesma estrutura do forecast, focada apenas na janela de teste (valor real + modelos).",
                style={
                    "color": T["text_muted"], "fontSize": "12px",
                    "textAlign": "center", "margin": "0 0 10px",
                    "lineHeight": "1.6",
                },
            ),
            dcc.Loading(
                dcc.Graph(
                    id="dash-forecast-test",
                    figure=empty_figure("Escolha uma página para ver o Forecast de Teste"),
                    config={"displayModeBar": False, "scrollZoom": False},
                    style={"height": "520px"},
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
            card([
                html.H4("Prophet", style={
                    "color": T["prophet_color"], "margin": "0 0 12px",
                    "fontSize": "14px", "fontWeight": "600",
                }),
                dcc.Loading(
                    html.Div(id="dash-prophet-metrics"),
                    type="dot", color=T["prophet_color"],
                ),
            ], style={"flex": "1"}),
            card([
                html.H4("AutoARIMA", style={
                    "color": T["auto_arima_color"], "margin": "0 0 12px",
                    "fontSize": "14px", "fontWeight": "600",
                }),
                dcc.Loading(
                    html.Div(id="dash-auto-arima-metrics"),
                    type="dot", color=T["auto_arima_color"],
                ),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px"}),

        # Scenario comparison section
        card([
            html.H4("Comparacao de Cenarios (3m, 6m, 9m)", style={
                "color": T["text"], "margin": "0 0 10px", "fontSize": "16px", "fontWeight": "700",
            }),
            html.P(
                "Compara o desempenho medio dos modelos em janelas diferentes de teste.",
                style={
                    "color": T["text_muted"], "fontSize": "12px",
                    "margin": "0 0 12px", "lineHeight": "1.6",
                },
            ),
            dcc.Loading(
                dcc.Graph(
                    id="dash-scenario-chart",
                    figure=empty_figure("Carregando comparacao de cenarios..."),
                    config={"displayModeBar": False, "scrollZoom": False},
                    style={"height": "320px"},
                ),
                type="dot", color=T["arima_color"],
            ),
            html.Div(id="dash-scenario-table"),
        ], style={"marginTop": "16px"}),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("dash-historical", "figure"),
    Input("page-selector", "value"),
)
def update_historical(page):
    if not page:
        return empty_figure("Selecione uma página para ver o histórico")
    data = fetch_series(page)
    if not data:
        return empty_figure("⚠ Não foi possível carregar os dados da série")
    return historical_chart(data["dates"], data["views"], page)


@callback(
    Output("dash-forecast-train", "figure"),
    Output("dash-status", "children"),
    Input("page-selector", "value"),
)
def update_forecast_train(page):
    if not page:
        return empty_figure("Escolha uma página para ver o Forecast de Treino"), ""
    train_fit = fetch_train_fit_forecasts(page)
    if not train_fit:
        return empty_figure("⏳ Aguarde — calculando fitted de treino dos modelos..."), \
               "Fitted de treino ainda não disponível para esta página."
    fig = train_fitted_models_chart(train_fit, page)

    all_fc = fetch_all_horizon_forecasts(page, "all")
    horizons = sorted({h for fc in all_fc.values() for h in fc})
    status = f"✓ {page.replace('_', ' ')} — Forecast Treino e Teste carregados — horizontes: {', '.join(str(h)+'d' for h in horizons)}"
    return fig, status


@callback(
    Output("dash-forecast-test", "figure"),
    Input("page-selector", "value"),
)
def update_forecast_test(page):
    if not page:
        return empty_figure("Escolha uma página para ver o Forecast de Teste")
    all_fc = fetch_all_horizon_forecasts(page, "all")
    if not all_fc:
        return empty_figure("⏳ Aguarde — forecasts sendo computados na inicialização...")
    data = fetch_series(page)
    if not data:
        return empty_figure("⚠ Não foi possível carregar os dados da série")
    return multi_horizon_test_chart(data["dates"], data["views"], all_fc, page)


@callback(
    Output("dash-arima-metrics", "children"),
    Output("dash-xgb-metrics", "children"),
    Output("dash-prophet-metrics", "children"),
    Output("dash-auto-arima-metrics", "children"),
    Input("page-selector", "value"),
)
def update_metrics(page):
    if not page:
        return _placeholder(), _placeholder(), _placeholder(), _placeholder()
    metrics = fetch_page_metrics(page)
    arima_m = next((m for m in metrics if m["model_type"] == "arima"), None)
    xgb_m = next((m for m in metrics if m["model_type"] == "xgboost"), None)
    prophet_m = next(
        (m for m in metrics if m["model_type"] == "prophet"), None
    )
    auto_arima_m = next((m for m in metrics if m["model_type"] == "auto_arima"), None)
    return (
        _render_metric_card(arima_m),
        _render_metric_card(xgb_m),
        _render_metric_card(prophet_m),
        _render_metric_card(auto_arima_m),
    )


@callback(
    Output("dash-scenario-chart", "figure"),
    Output("dash-scenario-table", "children"),
    Input("page-selector", "value"),
)
def update_scenario_compare(page):
    data = fetch_scenario_comparison(page)
    summary = data.get("summary", [])
    if not summary:
        return empty_figure("Cenarios ainda nao pre-computados"), html.Div(
            "Sem dados de comparacao de cenarios.",
            style={"color": T["text_muted"], "fontSize": "12px", "marginTop": "8px"},
        )

    fig = go.Figure()
    scenarios = sorted({r["scenario"] for r in summary}, key=lambda s: int(s.rstrip("m")))
    models = sorted({r["model_type"] for r in summary})
    for model in models:
        model_rows = [r for r in summary if r["model_type"] == model]
        mae_by_s = {r["scenario"]: r.get("mae") for r in model_rows}
        fig.add_trace(go.Scatter(
            x=scenarios,
            y=[mae_by_s.get(s) for s in scenarios],
            mode="lines+markers",
            name=model.upper(),
            line=dict(color={
                "arima": T["arima_color"],
                "xgboost": T["xgb_color"],
                "prophet": T["prophet_color"],
                "auto_arima": T["auto_arima_color"],
            }.get(model, T["text_muted"]), width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{model.upper()}</b><br>cenario: %{{x}}<br>MAE: %{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="MAE medio por cenario de teste", font=dict(color=T["text"], size=15)),
        paper_bgcolor=T["surface"],
        plot_bgcolor=T["surface"],
        font=dict(color=T["text"]),
        xaxis=dict(showgrid=False, linecolor=T["border"], tickfont=dict(color=T["text_muted"])),
        yaxis=dict(showgrid=False, linecolor=T["border"], tickfont=dict(color=T["text_muted"]), title="MAE"),
        legend=dict(bgcolor=T["surface2"], bordercolor=T["border"], borderwidth=1),
        margin=dict(l=50, r=20, t=45, b=40),
    )

    header = html.Thead(html.Tr([
        html.Th("Cenario", style=_th_style()),
        html.Th("Modelo", style=_th_style()),
        html.Th("MAE", style=_th_style()),
        html.Th("RMSE", style=_th_style()),
        html.Th("MAPE", style=_th_style()),
    ]))
    body_rows = [
        html.Tr([
            html.Td(r["scenario"], style=_td_style()),
            html.Td(r["model_type"].upper(), style=_td_style()),
            html.Td(f"{r.get('mae', float('nan')):.2f}", style=_td_style()),
            html.Td(f"{r.get('rmse', float('nan')):.2f}", style=_td_style()),
            html.Td(f"{r.get('mape', float('nan')):.2f}", style=_td_style()),
        ])
        for r in sorted(summary, key=lambda x: (x["scenario_months"], x.get("mae", 1e9)))
    ]
    table = html.Table(
        [header, html.Tbody(body_rows)],
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "marginTop": "10px",
            "fontSize": "12px",
        },
    )
    return fig, table


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


def _th_style():
    return {
        "textAlign": "left",
        "padding": "8px 10px",
        "borderBottom": f"1px solid {T['border']}",
        "color": T["text_muted"],
        "fontWeight": "600",
    }


def _td_style():
    return {
        "padding": "8px 10px",
        "borderBottom": f"1px solid {T['border']}",
        "color": T["text"],
    }
