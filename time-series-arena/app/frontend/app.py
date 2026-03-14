import dash
from dash import Dash, html, dcc
import flask

from config import THEME


T = THEME

_NAV = [
    ("Dashboard", "/"),
    ("Modelos", "/models"),
    ("Sobre", "/about"),
]


def create_app() -> Dash:
    app = Dash(
        __name__,
        use_pages=True,
        suppress_callback_exceptions=True,
        title="Time Series Arena",
        update_title=None,
    )

    # Callable layout — Dash calls this on every /_dash-layout request.
    # We reconstruct the page_container manually so we can pre-populate
    # _pages_content.children with the current page's layout.
    # This is the Dash 4.0 workaround for prevent_initial_call=True on the
    # built-in routing callback (content would otherwise be blank on first load).
    _NAV_H  = "48px"   # navbar height
    _FOOT_H = "34px"   # footer height

    def _layout():
        return html.Div([
            # ── Fixed Navbar ──────────────────────────────────────────────────
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
                "padding": f"0 24px",
                "height": _NAV_H,
                "display": "flex",
                "alignItems": "center",
                "position": "fixed",
                "top": "0",
                "left": "0",
                "right": "0",
                "zIndex": "1000",
            }),

            # ── Page content (offset for fixed navbar + footer) ───────────────
            html.Div([
                dcc.Location(id="_pages_location", refresh="callback-nav"),
                html.Div(
                    _initial_page_layout(),
                    id="_pages_content",
                    disable_n_clicks=True,
                    style={"paddingTop": _NAV_H, "paddingBottom": _FOOT_H},
                ),
                dcc.Store(id="_pages_store"),
                html.Div(id="_pages_dummy", disable_n_clicks=True),
            ]),

            # ── Fixed Footer ──────────────────────────────────────────────────
            html.Footer(
                html.Span(
                    "⚔️ Time Series Arena · Wikipedia pageview forecasting · ARIMA vs XGBoost",
                    style={"color": T["text_muted"], "fontSize": "12px"},
                ),
                style={
                    "backgroundColor": T["surface"],
                    "borderTop": f"1px solid {T['border']}",
                    "height": _FOOT_H,
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "position": "fixed",
                    "bottom": "0",
                    "left": "0",
                    "right": "0",
                    "zIndex": "1000",
                },
            ),
        ], style={
            "backgroundColor": T["bg"],
            "minHeight": "100vh",
            "fontFamily": "Inter, system-ui, sans-serif",
            "color": T["text"],
        })

    app.layout = _layout

    return app


def _initial_page_layout():
    """Return the page layout for the current request URL.

    Called inside a Flask request context (/_dash-layout).
    Falls back to the root page ('/') when no match is found.
    """
    try:
        from dash._pages import PAGE_REGISTRY, _path_to_page

        pathname = flask.request.path.rstrip("/") or "/"
        path_id = pathname.lstrip("/")
        page, path_variables = _path_to_page(path_id)

        if not page:
            # Try the root page as a fallback
            for p in PAGE_REGISTRY.values():
                if p.get("path") == "/":
                    page = p
                    path_variables = {}
                    break

        if page:
            layout = page.get("layout", "")
            if callable(layout):
                layout = layout(**(path_variables or {}))
            return layout
    except Exception:
        pass
    return None


def _link_style() -> dict:
    return {
        "color": T["text_muted"],
        "textDecoration": "none",
        "marginRight": "28px",
        "fontSize": "14px",
        "fontWeight": "500",
    }
