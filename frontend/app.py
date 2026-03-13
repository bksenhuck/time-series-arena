import dash
from dash import Dash, dcc, html

from frontend.config import THEME

T = THEME

_NAV = [
    ("Dashboard", "/"),
    ("Compare",   "/compare"),
    ("Dataset",   "/dataset"),
    ("Metrics",   "/metrics"),
]


def create_app() -> Dash:
    app = Dash(
        __name__,
        use_pages=True,
        suppress_callback_exceptions=True,
        title="Time Series Arena",
        update_title=None,
    )

    app.layout = html.Div([
        # ── Navigation ────────────────────────────────────────────────────────
        html.Nav([
            html.Div([
                html.Span("⚔️ Time Series Arena", style={
                    "color": T["text"],
                    "fontWeight": "700",
                    "fontSize": "18px",
                    "marginRight": "40px",
                    "letterSpacing": "-0.3px",
                    "userSelect": "none",
                }),
                *[
                    dcc.Link(label, href=href, style=_link_style())
                    for label, href in _NAV
                ],
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={
            "backgroundColor": T["surface"],
            "borderBottom": f"1px solid {T['border']}",
            "padding": "12px 24px",
            "position": "sticky",
            "top": "0",
            "zIndex": "100",
        }),

        # ── Page content ──────────────────────────────────────────────────────
        dash.page_container,
    ], style={
        "backgroundColor": T["bg"],
        "minHeight": "100vh",
        "fontFamily": "Inter, system-ui, sans-serif",
        "color": T["text"],
    })

    return app


def _link_style() -> dict:
    return {
        "color": T["text_muted"],
        "textDecoration": "none",
        "marginRight": "28px",
        "fontSize": "14px",
        "fontWeight": "500",
    }
