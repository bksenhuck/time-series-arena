"""About page describing concept, methodology, models, and roadmap."""
import dash
from dash import html

from app.frontend.components.layout import card, page_container, page_header
from config import THEME

dash.register_page(__name__, path="/about", title="Sobre")

T = THEME


def _bullet(text: str) -> html.Li:
    return html.Li(
        text,
        style={"color": T["text"], "fontSize": "14px", "lineHeight": "1.7", "marginBottom": "6px"},
    )


def _card_title(text: str) -> html.H3:
    return html.H3(
        text,
        style={
            "color": T["text"],
            "margin": "0 0 12px",
            "fontSize": "18px",
            "fontWeight": "700",
            "letterSpacing": "-0.2px",
        },
    )


def layout() -> html.Div:
    return page_container([
        page_header(
            "Sobre o Time Series Arena",
            "Visao geral do conceito, metodologia, modelos e proximos passos do projeto.",
        ),

        card([
            _card_title("Conceito"),
            html.Ul([
                _bullet("O Time Series Arena compara modelos de previsao em uma base real de pageviews da Wikipedia."),
                _bullet("A proposta e separar treino e teste para avaliar generalizacao, nao apenas ajuste visual."),
                _bullet("O dashboard mostra historico, ajuste no treino e comportamento em teste com o mesmo padrao visual."),
                _bullet("A ideia central e permitir comparacao transparente de desempenho por modelo e por horizonte."),
            ], style={"margin": "0", "paddingLeft": "20px"}),
        ], style={"marginBottom": "16px"}),

        card([
            _card_title("Metodologia"),
            html.Ul([
                _bullet("Coleta: janela configuravel de dados com DATA_MIN_DATE e DATA_MAX_DATE."),
                _bullet("Split movel: TEST_WINDOW_MONTHS define quantos meses finais ficam em teste."),
                _bullet("Treino: modelos ajustados apenas no periodo de treino para evitar vazamento."),
                _bullet("Avaliacao: metricas walk-forward e comparacao visual entre real e previsto."),
                _bullet("Pre-computacao: forecasts por multiplos horizontes para resposta rapida no dashboard."),
            ], style={"margin": "0", "paddingLeft": "20px"}),
        ], style={"marginBottom": "16px"}),

        card([
            _card_title("Modelos"),
            html.Ul([
                _bullet("ARIMA: baseline estatistico local, robusto para padroes lineares e curtos horizontes."),
                _bullet("Prophet: tendencia + sazonalidade com boa interpretabilidade em series com ciclos."),
                _bullet("XGBoost: modelo global com lags e calendario, forte em nao linearidades."),
                _bullet("AutoARIMA: selecao automatica de ordem para reduzir tuning manual."),
                _bullet("Cada modelo possui pontos fortes diferentes, por isso a comparacao lado a lado e essencial."),
            ], style={"margin": "0", "paddingLeft": "20px"}),
        ], style={"marginBottom": "16px"}),

        card([
            _card_title("Proximos Passos"),
            html.Ul([
                _bullet("Backtest rolling para preencher toda a janela de teste com previsoes sequenciais."),
                _bullet("Calibracao de intervalos de confianca para comparacao de risco por modelo."),
                _bullet("Relatorios automaticos com ranking consolidado por pagina e por horizonte."),
                _bullet("Deteccao de drift para sinalizar quando o modelo precisa de retreino."),
                _bullet("Suporte a novas fontes de serie temporal alem de pageviews."),
            ], style={"margin": "0", "paddingLeft": "20px"}),
        ]),
    ])
