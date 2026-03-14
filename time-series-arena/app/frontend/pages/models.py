"""Models page — brief comparison of the four forecasting models."""
import dash
from dash import html

from app.frontend.components.layout import card, page_container, page_header
from config import THEME

dash.register_page(__name__, path="/models", title="Modelos")

T = THEME


def _pro(text: str) -> html.Li:
    return html.Li(text, style={
        "color": T["text"], "fontSize": "13px",
        "lineHeight": "1.7", "marginBottom": "4px",
    })


def _con(text: str) -> html.Li:
    return html.Li(text, style={
        "color": T["text_muted"], "fontSize": "13px",
        "lineHeight": "1.7", "marginBottom": "4px",
    })


def _section_label(text: str, color: str) -> html.Div:
    return html.Div(text, style={
        "color": color,
        "fontSize": "11px",
        "fontWeight": "700",
        "textTransform": "uppercase",
        "letterSpacing": "0.6px",
        "marginBottom": "6px",
        "marginTop": "14px",
    })


def _impl_item(text: str) -> html.Li:
    return html.Li(text, style={
        "color": T["text"], "fontSize": "12px",
        "lineHeight": "1.7", "marginBottom": "3px",
        "fontFamily": "monospace",
    })


def _model_card(
    name: str,
    color: str,
    summary: str,
    impl: list[str],
    pros: list[str],
    cons: list[str],
) -> html.Div:
    return card([
        html.Div([
            html.H3(name, style={
                "color": color, "margin": "0",
                "fontSize": "17px", "fontWeight": "700",
            }),
        ], style={"marginBottom": "8px"}),

        html.P(summary, style={
            "color": T["text_muted"], "fontSize": "13px",
            "lineHeight": "1.6", "margin": "0",
        }),

        _section_label("Neste projeto", color),
        html.Div([
            html.Ul([_impl_item(i) for i in impl], style={"margin": "0", "paddingLeft": "18px"}),
        ], style={
            "backgroundColor": T["bg"],
            "border": f"1px solid {T['border']}",
            "borderRadius": "6px",
            "padding": "10px 12px",
        }),

        _section_label("Vantagens", color),
        html.Ul([_pro(p) for p in pros], style={"margin": "0", "paddingLeft": "18px"}),

        _section_label("Desvantagens", T["text_muted"]),
        html.Ul([_con(c) for c in cons], style={"margin": "0", "paddingLeft": "18px"}),

    ], style={"flex": "1", "minWidth": "260px", "marginBottom": "16px"})


def layout() -> html.Div:
    return page_container([
        page_header(
            "Modelos de Forecasting",
            "Comparativo rapido entre os quatro modelos usados no Time Series Arena.",
        ),

        # Grid 2x2
        html.Div([
            _model_card(
                name="ARIMA",
                color=T["arima_color"],
                summary=(
                    "Modelo estatistico local que modela autocorrelacao linear via "
                    "termos autorregressivos (AR), diferenciacao (I) e media movel (MA). "
                    "Treinado individualmente por serie."
                ),
                impl=[
                    "Local: re-treinado por serie a cada predict(), sem estado compartilhado",
                    "Ordem fixa: SARIMA(2,1,2)(1,0,1,7) — inclui sazonalidade semanal",
                    "Fallbacks automaticos para (1,1,1) se a ordem padrao divergir",
                    "Intervalo de confianca direto do SARIMAX (alpha=0.05)",
                ],
                pros=[
                    "Forte em series estacionarias e horizontes curtos",
                    "Interpretavel: parametros p, d, q tem significado claro",
                    "Baixo custo computacional para series unicas",
                    "Solido como baseline estatistico",
                ],
                cons=[
                    "Nao captura nao linearidades",
                    "Ordem fixa pode nao ser otima para todas as series",
                    "Perde qualidade rapidamente em horizontes longos",
                    "Nao aproveita padroes entre series diferentes",
                ],
            ),
            _model_card(
                name="Auto-ARIMA",
                color=T["auto_arima_color"],
                summary=(
                    "Variante automatizada do ARIMA que seleciona (p, d, q) por busca "
                    "stepwise minimizando AIC/BIC. Reduz o tuning manual sem alterar "
                    "a estrutura do modelo."
                ),
                impl=[
                    "Local: tambem re-treinado por serie a cada predict()",
                    "Busca stepwise de (p,d,q) via pmdarima minimizando AIC",
                    "Mesma estrutura de previsao do ARIMA apos selecao de ordem",
                    "Intervalo de confianca via SARIMAX apos ordem selecionada",
                ],
                pros=[
                    "Elimina tuning manual de ordem",
                    "Selecao principiada por criterio de informacao (AIC/BIC)",
                    "Boa referencia para comparar com ARIMA fixo",
                    "Robusto para series com diferentes graus de integracao",
                ],
                cons=[
                    "Mais lento que ARIMA fixo (busca em grid)",
                    "Mesmas limitacoes estruturais do ARIMA (linearidade)",
                    "Pode selecionar ordem sub-otima em series curtas",
                    "Sem sazonalidade automatica se nao configurado",
                ],
            ),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        html.Div([
            _model_card(
                name="Prophet",
                color=T["prophet_color"],
                summary=(
                    "Modelo aditivo do Meta que decompoe a serie em tendencia, "
                    "sazonalidade (anual, semanal) e feriados. Projetado para "
                    "series com ciclos regulares e descontinuidades."
                ),
                impl=[
                    "Local: treinado individualmente por serie a cada predict()",
                    "Componentes: tendencia + sazonalidade semanal e anual",
                    "Previsao multi-step direta (gera todos os passos de uma vez)",
                    "Intervalo de credibilidade via incerteza do modelo aditivo",
                ],
                pros=[
                    "Captura sazonalidade multipla nativamente",
                    "Robusto a dados faltantes e outliers",
                    "Interpretavel: componentes separados de tendencia e sazonalidade",
                    "Incerteza quantificada com intervalos de credibilidade",
                ],
                cons=[
                    "Mais lento para treinar que ARIMA",
                    "Hipoteses aditivas podem nao valer para series volateis",
                    "Menos preciso em series sem sazonalidade clara",
                    "Horizonte muito longo tende a suavizar demais",
                ],
            ),
            _model_card(
                name="XGBoost",
                color=T["xgb_color"],
                summary=(
                    "Modelo global de gradient boosting treinado com lags e features "
                    "de calendario como variaveis de entrada. Aprende padroes "
                    "simultaneamente de todas as series."
                ),
                impl=[
                    "Global: unico modelo treinado em todas as 25 series simultaneamente",
                    "Features: lag_1/7/14/30, rolling_mean_7/30, day_of_week, month, series_id_enc",
                    "Previsao recursiva: previsoes anteriores viram lags nos proximos passos",
                    "Diferencia series via LabelEncoder no series_id",
                ],
                pros=[
                    "Captura nao linearidades e interacoes complexas",
                    "Aproveita padroes entre series (modelo global)",
                    "Extensivel com qualquer feature externa",
                    "Alta capacidade preditiva em horizontes medios",
                ],
                cons=[
                    "Caixa preta: menor interpretabilidade",
                    "Feature engineering manual (lags, calendario)",
                    "Requer mais dados para generalizar bem",
                    "Extrapolacao fraca alem do range de treino",
                ],
            ),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # Summary table
        card([
            html.H3("Quando usar cada um", style={
                "color": T["text"], "margin": "0 0 16px",
                "fontSize": "16px", "fontWeight": "700",
            }),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Situacao", style=_th()),
                    html.Th("Recomendado", style=_th()),
                ])),
                html.Tbody([
                    _row("Baseline rapido, serie curta",          "ARIMA"),
                    _row("Sem tempo para tuning de ordem",        "Auto-ARIMA"),
                    _row("Serie com ciclos semanais/anuais claros","Prophet"),
                    _row("Muitas series, features externas",      "XGBoost"),
                    _row("Comparacao justa entre abordagens",     "Todos — e o objetivo do Arena"),
                ]),
            ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "13px"}),
        ]),
    ])


def _th() -> dict:
    return {
        "textAlign": "left",
        "padding": "8px 12px",
        "borderBottom": f"1px solid {T['border']}",
        "color": T["text_muted"],
        "fontWeight": "600",
        "fontSize": "12px",
        "textTransform": "uppercase",
        "letterSpacing": "0.4px",
    }


def _row(situation: str, model: str) -> html.Tr:
    color_map = {
        "ARIMA": T["arima_color"],
        "Auto-ARIMA": T["auto_arima_color"],
        "Prophet": T["prophet_color"],
        "XGBoost": T["xgb_color"],
    }
    model_color = color_map.get(model, T["text"])
    return html.Tr([
        html.Td(situation, style={
            "padding": "9px 12px",
            "borderBottom": f"1px solid {T['border']}",
            "color": T["text"],
        }),
        html.Td(model, style={
            "padding": "9px 12px",
            "borderBottom": f"1px solid {T['border']}",
            "color": model_color,
            "fontWeight": "600",
        }),
    ])
