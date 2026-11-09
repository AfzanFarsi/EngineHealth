"""Look & feel of the website: CSS + small HTML components."""
from __future__ import annotations

import html

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
body, .stMarkdown, .stMarkdown p, h1, h2, h3, label { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
.stApp { background: radial-gradient(1200px 500px at 85% -10%, rgba(34,211,238,.08), transparent 60%),
                     radial-gradient(900px 500px at -10% 0%, rgba(59,130,246,.10), transparent 55%), #0B1220; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: #0E1629; border-right: 1px solid #1B2740; }
h1, h2, h3 { letter-spacing: -0.01em; }
h1 { font-weight: 700; } h2, h3 { font-weight: 600; }

.eh-hero { padding: 3.2rem 2.6rem; border-radius: 22px; border: 1px solid #1F2D4A;
  background: linear-gradient(135deg, rgba(34,211,238,.10), rgba(59,130,246,.10) 45%, rgba(18,27,46,.9));
  position: relative; overflow: hidden; margin-bottom: 1.6rem; }
.eh-hero:after { content:""; position:absolute; right:-80px; top:-80px; width:340px; height:340px; border-radius:50%;
  background: radial-gradient(circle, rgba(34,211,238,.22), transparent 65%); }
.eh-eyebrow { color:#22D3EE; font-size:.78rem; letter-spacing:.16em; font-weight:600; text-transform:uppercase; }
.eh-hero h1 { font-size: 3rem; line-height:1.08; margin:.5rem 0 .8rem 0; max-width: 760px; }
.eh-grad { background: linear-gradient(90deg,#22D3EE,#3B82F6); -webkit-background-clip:text; background-clip:text; color:transparent; }
.eh-lead { color:#A9B7D0; font-size:1.08rem; max-width: 720px; line-height:1.6; }

.eh-card { background:#121B2E; border:1px solid #1F2D4A; border-radius:16px; padding:1.15rem 1.3rem; height:100%;
  transition: border-color .2s, transform .2s; }
.eh-card:hover { border-color:#2B6CB0; transform: translateY(-2px); }
.eh-card h4 { margin:.15rem 0 .35rem 0; font-size:1.02rem; color:#E6EDF7; }
.eh-card p { color:#9FB0CB; font-size:.9rem; line-height:1.55; margin:0; }
.eh-icon { width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center;
  background: rgba(34,211,238,.12); color:#22D3EE; font-size:1.35rem; margin-bottom:.6rem;
  font-family:'Material Symbols Rounded'; font-weight:400; font-variation-settings:'FILL' 0,'wght' 400; }

.eh-kpi { background:#121B2E; border:1px solid #1F2D4A; border-radius:16px; padding:1rem 1.2rem; position:relative; overflow:hidden; }
.eh-kpi:before { content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background: var(--accent,#22D3EE); }
.eh-kpi .l { color:#8FA0BC; font-size:.76rem; letter-spacing:.06em; text-transform:uppercase; font-weight:600; }
.eh-kpi .v { font-size:1.85rem; font-weight:700; margin-top:.15rem; color:#E6EDF7; }
.eh-kpi .s { color:#8FA0BC; font-size:.78rem; margin-top:.1rem; }

.eh-badge { display:inline-block; padding:.2rem .7rem; border-radius:999px; font-size:.78rem; font-weight:600; border:1px solid; }
.eh-Healthy { color:#34D399; background:rgba(52,211,153,.12); border-color:rgba(52,211,153,.4); }
.eh-Warning { color:#FBBF24; background:rgba(251,191,36,.12); border-color:rgba(251,191,36,.4); }
.eh-Critical { color:#F87171; background:rgba(248,113,113,.12); border-color:rgba(248,113,113,.4); }
.eh-info { color:#22D3EE; background:rgba(34,211,238,.10); border-color:rgba(34,211,238,.35); }

.eh-section { margin: 1.6rem 0 .6rem 0; }
.eh-section h2 { margin:0; font-size:1.35rem; }
.eh-section p { margin:.2rem 0 0 0; color:#8FA0BC; font-size:.92rem; }
.eh-note { border-left:3px solid #3B82F6; background:rgba(59,130,246,.08); padding:.7rem 1rem; border-radius:0 10px 10px 0;
  color:#B6C4DD; font-size:.9rem; margin:.5rem 0 1rem 0; }
.eh-step { display:flex; gap:.9rem; align-items:flex-start; }
.eh-step .n { min-width:34px; height:34px; border-radius:50%; background:linear-gradient(135deg,#22D3EE,#3B82F6); color:#06101F;
  font-weight:700; display:flex; align-items:center; justify-content:center; }
.eh-pill { display:inline-block; background:#16223A; border:1px solid #243556; color:#BFD0EA; padding:.25rem .7rem;
  border-radius:8px; font-size:.82rem; margin:.15rem .25rem .15rem 0; }
div[data-testid="stTabs"] button[role="tab"] { font-weight:600; }
div[data-testid="stDataFrame"] { border:1px solid #1F2D4A; border-radius:12px; overflow:hidden; }
.stButton > button, .stDownloadButton > button { border-radius:10px; font-weight:600; border:1px solid #2B6CB0; }
.stButton > button[kind="primary"] { background: linear-gradient(90deg,#06B6D4,#3B82F6); border:none; color:#04111F; }
@media (max-width: 768px) { .eh-hero { padding: 2rem 1.3rem; } .eh-hero h1 { font-size: 2.1rem; } }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def section(title: str, subtitle: str = "") -> None:
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    st.markdown(f"<div class='eh-section'><h2>{esc(title)}</h2>{sub}</div>", unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<div class='eh-note'>{text}</div>", unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", accent: str = "#22D3EE") -> None:
    st.markdown(f"<div class='eh-kpi' style='--accent:{accent}'><div class='l'>{esc(label)}</div>"
                f"<div class='v'>{esc(value)}</div><div class='s'>{esc(sub)}</div></div>", unsafe_allow_html=True)


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        with c:
            kpi(*it)


def badge(text: str, kind: str = "info") -> str:
    return f"<span class='eh-badge eh-{esc(kind)}'>{esc(text)}</span>"


def card(icon: str, title: str, body: str) -> str:
    return (f"<div class='eh-card'><div class='eh-icon'>{icon}</div><h4>{esc(title)}</h4><p>{esc(body)}</p></div>")


def pills(items: list[str]) -> str:
    return "".join(f"<span class='eh-pill'>{esc(i)}</span>" for i in items)
