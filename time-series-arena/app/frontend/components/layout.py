"""Shared layout primitives used across all pages."""
from dash import html

from config import THEME

T = THEME


# ── Base building-blocks ──────────────────────────────────────────────────────

def card(children, style: dict | None = None) -> html.Div:
    """Dark surface card with a subtle border."""
    base = {
        "backgroundColor": T["surface"],
        "border": f"1px solid {T['border']}",
        "borderRadius": "10px",
        "padding": "20px",
    }
    return html.Div(children, style={**base, **(style or {})})


def page_container(children, max_width: str = "1400px") -> html.Div:
    """Full-page wrapper with consistent padding and max-width."""
    return html.Div(
        html.Div(
            children,
            style={"maxWidth": max_width, "margin": "0 auto"},
        ),
        style={
            "padding": "28px 24px",
            "backgroundColor": T["bg"],
            "minHeight": "100vh",
        },
    )


def page_header(title: str, subtitle: str = "") -> html.Div:
    children: list = [
        html.H2(title, style={
            "color": T["text"],
            "margin": "0",
            "fontSize": "22px",
            "fontWeight": "700",
            "letterSpacing": "-0.3px",
        }),
    ]
    if subtitle:
        children.append(
            html.P(subtitle, style={
                "color": T["text_muted"],
                "margin": "4px 0 0",
                "fontSize": "13px",
            })
        )
    return html.Div(children, style={"marginBottom": "24px"})


def control_label(text: str) -> html.Label:
    return html.Label(text, style={
        "color": T["text_muted"],
        "fontSize": "11px",
        "fontWeight": "600",
        "textTransform": "uppercase",
        "letterSpacing": "0.5px",
        "marginBottom": "6px",
        "display": "block",
    })


def stat_card(
    title: str,
    color: str,
    stats: list[tuple[str, str]],
    footnote: str = "",
) -> html.Div:
    """KPI card showing a title, a list of (label, value) pairs, and optional footnote."""
    rows = [
        html.Div([
            html.Div(label, style={
                "color": T["text_muted"],
                "fontSize": "11px",
                "fontWeight": "600",
                "textTransform": "uppercase",
                "letterSpacing": "0.3px",
            }),
            html.Div(value, style={
                "color": T["text"],
                "fontSize": "24px",
                "fontWeight": "700",
                "lineHeight": "1.2",
            }),
        ], style={"marginBottom": "10px"})
        for label, value in stats
    ]
    footer = [html.Div(footnote, style={
        "color": T["text_muted"], "fontSize": "11px", "marginTop": "12px",
        "borderTop": f"1px solid {T['border']}", "paddingTop": "8px",
    })] if footnote else []
    return card(
        [html.H4(title, style={
            "color": color,
            "margin": "0 0 14px",
            "fontSize": "13px",
            "fontWeight": "700",
            "textTransform": "uppercase",
            "letterSpacing": "0.5px",
        })] + rows + footer,
        style={"flex": "1", "minWidth": "180px"},
    )


def divider() -> html.Hr:
    return html.Hr(style={"border": "none",
                          "borderTop": f"1px solid {T['border']}",
                          "margin": "20px 0"})
