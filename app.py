"""Streamlit dashboard for jewel image matching using DinoV2."""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from jewel_matcher import (
    EMBEDDING_DIM,
    MODEL_NAME,
    detect_device,
    load_model,
    match,
)

st.set_page_config(
    page_title="Emerald-Gold-Ring-Classification",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500;1,600;1,700&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --ink: #F4F1EA;
    --ink-soft: #B8B0A4;
    --ink-muted: #6B635A;
    --bg: #0A0908;
    --bg-elevated: #14110F;
    --bg-card: #1A1714;
    --gold: #C9A961;
    --gold-bright: #E0C078;
    --gold-soft: rgba(201, 169, 97, 0.18);
    --gold-faint: rgba(201, 169, 97, 0.06);
    --crimson: #B23C4B;
    --hairline: rgba(244, 241, 234, 0.08);
    --hairline-strong: rgba(244, 241, 234, 0.18);
}

#MainMenu, footer, [data-testid="stDeployButton"], [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stStatusWidget"] {
    display: none !important;
    visibility: hidden !important;
}

[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}

html, body, [data-testid="stAppViewContainer"], .stApp, .main {
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family: 'Instrument Sans', -apple-system, sans-serif;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background:
        radial-gradient(ellipse 70% 35% at 50% -10%, rgba(201, 169, 97, 0.10) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 110% 110%, rgba(178, 60, 75, 0.06) 0%, transparent 70%);
    z-index: 0;
}

.block-container {
    max-width: 1280px;
    padding-top: 3rem !important;
    padding-bottom: 6rem !important;
    position: relative;
    z-index: 1;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Cormorant Garamond', serif !important;
    color: var(--ink) !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
}

p, span, label, div, li {
    color: var(--ink) !important;
}

[data-testid="stMarkdownContainer"] p {
    font-family: 'Instrument Sans', sans-serif;
    line-height: 1.65;
    color: var(--ink-soft) !important;
}

/* ====== MASTHEAD ====== */
.masthead {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--hairline-strong);
    margin-bottom: 4rem;
    font-family: 'Instrument Sans', sans-serif;
    font-size: 0.68rem;
    letter-spacing: 0.32em;
    text-transform: uppercase;
}
.masthead .brand {
    color: var(--ink) !important;
    display: flex;
    align-items: center;
    gap: 0.85rem;
}
.masthead .brand .mark {
    color: var(--gold) !important;
    font-size: 1rem;
    letter-spacing: 0;
}
.masthead .meta {
    color: var(--ink-muted) !important;
    display: flex;
    gap: 1.75rem;
}
.masthead .meta .pulse {
    color: var(--gold) !important;
}
.masthead .meta .pulse::before {
    content: "●";
    color: var(--gold);
    margin-right: 0.45rem;
    animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.35; }
}

/* ====== HERO ====== */
.hero {
    padding-bottom: 3.5rem;
    border-bottom: 1px solid var(--hairline);
    margin-bottom: 3rem;
}
.hero-eyebrow {
    font-family: 'Instrument Sans', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.38em;
    text-transform: uppercase;
    color: var(--gold) !important;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 0.9rem;
}
.hero-eyebrow::before {
    content: "";
    width: 36px; height: 1px;
    background: var(--gold);
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 400;
    font-size: clamp(3.5rem, 7vw, 6.5rem);
    line-height: 0.92;
    letter-spacing: -0.035em;
    color: var(--ink) !important;
    margin: 0 0 1.5rem 0;
}
.hero-title .accent {
    color: var(--gold) !important;
    font-style: italic;
    font-weight: 500;
}
.hero-title .dot {
    color: var(--gold) !important;
}
.hero-sub {
    font-family: 'Instrument Sans', sans-serif;
    font-size: 1.02rem;
    line-height: 1.7;
    color: var(--ink-soft) !important;
    max-width: 580px;
    font-weight: 400;
}

/* ====== INFO PANEL ====== */
.info-panel {
    background: linear-gradient(180deg, var(--bg-elevated) 0%, rgba(20,17,15,0.6) 100%);
    border: 1px solid var(--hairline);
    padding: 2rem 1.75rem;
    position: relative;
    overflow: hidden;
}
.info-panel::before {
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 48px; height: 1px;
    background: var(--gold);
}
.info-panel::after {
    content: "◈";
    position: absolute;
    top: 1rem; right: 1.25rem;
    color: var(--gold);
    opacity: 0.4;
    font-size: 0.75rem;
}
.info-panel-title {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1.1rem;
    color: var(--ink) !important;
    margin-bottom: 1rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid var(--hairline);
}
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.7rem 0;
    border-bottom: 1px dotted var(--hairline);
    font-family: 'Instrument Sans', sans-serif;
}
.info-row:last-child { border-bottom: none; padding-bottom: 0; }
.info-key {
    font-size: 0.62rem !important;
    letter-spacing: 0.28em !important;
    text-transform: uppercase;
    color: var(--ink-muted) !important;
}
.info-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: var(--ink) !important;
    font-weight: 400;
}
.info-val.gold { color: var(--gold) !important; font-weight: 500; }

/* ====== SECTION LABELS ====== */
.section-label {
    font-family: 'Instrument Sans', sans-serif;
    font-size: 0.66rem;
    letter-spacing: 0.38em;
    text-transform: uppercase;
    color: var(--gold) !important;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.85rem;
}
.section-label .num {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1.4rem;
    color: var(--gold) !important;
    letter-spacing: normal;
    text-transform: none;
    line-height: 1;
}
.section-label .line {
    flex: 1;
    height: 1px;
    background: var(--hairline);
}

/* ====== FILE UPLOADER ====== */
[data-testid="stFileUploader"] {
    background: transparent;
}
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploader"] section {
    background: var(--bg-elevated) !important;
    border: 1px dashed var(--hairline-strong) !important;
    border-radius: 0 !important;
    padding: 2rem !important;
    transition: all 0.3s cubic-bezier(0.22, 1, 0.36, 1);
}
[data-testid="stFileUploaderDropzone"]:hover, [data-testid="stFileUploader"] section:hover {
    border-color: var(--gold) !important;
    background: var(--bg-card) !important;
    transform: translateY(-1px);
}
[data-testid="stFileUploader"] button, [data-testid="baseButton-secondary"] {
    background: transparent !important;
    border: 1px solid var(--gold) !important;
    color: var(--gold) !important;
    border-radius: 0 !important;
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.3rem !important;
    font-weight: 500 !important;
    transition: all 0.25s ease;
}
[data-testid="stFileUploader"] button:hover {
    background: var(--gold) !important;
    color: var(--bg) !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--ink-muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: var(--ink-soft) !important;
    font-family: 'Instrument Sans', sans-serif;
}
[data-testid="stFileUploaderFile"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--hairline) !important;
}
[data-testid="stFileUploaderFile"] * {
    color: var(--ink-soft) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ====== IMAGE ====== */
[data-testid="stImage"] img {
    border: 1px solid var(--hairline);
    border-radius: 0;
    filter: contrast(1.03) saturate(0.96);
}
[data-testid="stImage"] {
    margin-top: 1rem;
}

/* ====== SLIDER ====== */
[data-testid="stSlider"] {
    padding: 0.5rem 0 1.25rem 0;
}
[data-testid="stSlider"] label, [data-testid="stSlider"] label p {
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.32em !important;
    text-transform: uppercase !important;
    color: var(--ink-muted) !important;
    font-weight: 500 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
    background: var(--gold) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div {
    background: var(--hairline-strong) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: var(--gold) !important;
    box-shadow: 0 0 0 5px rgba(201, 169, 97, 0.16) !important;
    border: none !important;
    width: 14px !important;
    height: 14px !important;
}
[data-testid="stSlider"] [data-testid="stTickBar"] {
    color: var(--ink-muted) !important;
}
[data-testid="stSlider"] [data-testid="stTickBar"] * {
    color: var(--ink-muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
}
/* Slider current value bubble */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] + div,
[data-testid="stSlider"] div[aria-valuenow] {
    color: var(--gold) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ====== PRIMARY BUTTON ====== */
[data-testid="stButton"] button, [data-testid="baseButton-primary"] {
    background: transparent !important;
    color: var(--gold) !important;
    border: 1px solid var(--gold) !important;
    border-radius: 0 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-style: italic !important;
    font-size: 1.4rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.015em !important;
    padding: 1.15rem 2rem !important;
    transition: all 0.35s cubic-bezier(0.22, 1, 0.36, 1);
    position: relative;
    overflow: hidden;
}
[data-testid="stButton"] button::before {
    content: "";
    position: absolute;
    inset: 0;
    background: var(--gold);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.4s cubic-bezier(0.22, 1, 0.36, 1);
    z-index: -1;
}
[data-testid="stButton"] button:hover::before {
    transform: scaleX(1);
}
[data-testid="stButton"] button:hover,
[data-testid="stButton"] button:focus:not(:active) {
    background: var(--gold) !important;
    color: var(--bg) !important;
    box-shadow: none !important;
}

/* ====== METRIC CARD ====== */
.metric-card {
    padding-top: 1.25rem;
    border-top: 1px solid var(--hairline-strong);
    position: relative;
}
.metric-card::before {
    content: "";
    position: absolute;
    top: -1px; left: 0;
    width: 24px; height: 1px;
    background: var(--gold);
}
.metric-label {
    font-family: 'Instrument Sans', sans-serif;
    font-size: 0.6rem;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: var(--ink-muted) !important;
    margin-bottom: 0.85rem;
    font-weight: 500;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.5rem;
    font-weight: 300;
    color: var(--ink) !important;
    line-height: 1;
    letter-spacing: -0.02em;
}
.metric-value .unit {
    font-size: 0.85rem;
    color: var(--ink-muted) !important;
    margin-left: 0.4rem;
    font-weight: 400;
}
.metric-value.gold { color: var(--gold) !important; }

/* ====== VERDICT ====== */
.verdict {
    padding: 4rem 0 2.5rem 0;
    border-top: 1px solid var(--hairline);
    border-bottom: 1px solid var(--hairline);
    margin: 3.5rem 0 3rem 0;
    position: relative;
}
.verdict::before {
    content: "";
    position: absolute;
    top: -1px; left: 0;
    width: 80px; height: 1px;
    background: var(--gold);
}
.verdict-grid {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 2rem;
    align-items: end;
}
.verdict-eyebrow {
    font-family: 'Instrument Sans', sans-serif;
    font-size: 0.66rem;
    letter-spacing: 0.42em;
    text-transform: uppercase;
    color: var(--gold) !important;
    margin-bottom: 1.5rem;
}
.verdict-word {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-weight: 500;
    font-size: clamp(5rem, 11vw, 9.5rem);
    line-height: 0.88;
    letter-spacing: -0.045em;
    margin: 0;
    animation: verdict-rise 0.8s cubic-bezier(0.22, 1, 0.36, 1);
}
.verdict-word.pass { color: var(--gold) !important; }
.verdict-word.fail { color: var(--crimson) !important; }
@keyframes verdict-rise {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.verdict-aside {
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--ink-muted) !important;
    line-height: 1.9;
    letter-spacing: 0.04em;
    padding-bottom: 0.5rem;
}
.verdict-aside .num {
    color: var(--ink) !important;
    font-size: 1.05rem;
    font-weight: 400;
}
.verdict-aside .lbl {
    font-family: 'Instrument Sans', sans-serif;
    font-size: 0.58rem;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: var(--ink-muted) !important;
    display: block;
    margin-top: 0.4rem;
}

/* ====== STAGE INDICATOR ====== */
.stage {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1rem;
    color: var(--ink-muted) !important;
    text-align: center;
    padding: 3rem 0;
}
.stage::before, .stage::after {
    content: "—";
    color: var(--gold);
    margin: 0 1rem;
    font-style: normal;
}

/* ====== DIVIDER ====== */
.divider {
    height: 1px;
    background: var(--hairline);
    margin: 3.5rem 0;
    position: relative;
}
.divider::before {
    content: "◈";
    position: absolute;
    top: -10px;
    left: 50%;
    transform: translateX(-50%);
    background: var(--bg);
    padding: 0 1rem;
    color: var(--gold);
    font-size: 0.7rem;
}

/* ====== ANNOTATIONS ====== */
.annotation {
    font-family: 'Instrument Sans', sans-serif;
    font-size: 0.78rem;
    color: var(--ink-muted) !important;
    line-height: 1.7;
    margin-top: 1.25rem;
    padding-left: 1.25rem;
    border-left: 1px solid var(--gold);
    max-width: 720px;
    font-style: normal;
}

/* ====== PYPLOT ====== */
[data-testid="stPyplot"] {
    background: transparent;
    margin-top: 1rem;
}

/* ====== SPINNER ====== */
[data-testid="stSpinner"] {
    color: var(--gold) !important;
}
[data-testid="stSpinner"] > div, [data-testid="stSpinner"] i {
    border-top-color: var(--gold) !important;
    color: var(--gold) !important;
}

/* Hide unwanted streamlit elements */
.stDeployButton, .viewerBadge_container__1QSob {
    display: none !important;
}

</style>
"""


@st.cache_resource(show_spinner="◈  Loading vision transformer...")
def _get_model_and_processor(device: str):
    return load_model(device)


ONYX_GOLD = LinearSegmentedColormap.from_list(
    "onyx_gold",
    [
        (0.00, "#0A0908"),
        (0.25, "#2A1F12"),
        (0.55, "#6B4F22"),
        (0.82, "#C9A961"),
        (1.00, "#F4DCA0"),
    ],
)


def _style_axis(ax, bg: str) -> None:
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color("#1F1B17")
        spine.set_linewidth(0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _embedding_figure(template_emb: np.ndarray, input_emb: np.ndarray):
    bg = "#0A0908"
    gold = "#C9A961"

    diff = template_emb - input_emb
    diff_abs = np.abs(diff).reshape(24, 32)
    l2 = float(np.linalg.norm(diff))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), facecolor=bg)

    for ax, grid, title in zip(
        axes[:2],
        [template_emb.reshape(24, 32), input_emb.reshape(24, 32)],
        ["Template", "Input"],
    ):
        _style_axis(ax, bg)
        ax.imshow(grid, cmap="magma", aspect="equal", interpolation="bilinear")
        ax.set_title(
            title.upper(),
            color=gold,
            fontsize=8.5,
            family="serif",
            fontstyle="italic",
            pad=14,
            loc="left",
        )

    ax = axes[2]
    _style_axis(ax, bg)
    if l2 < 1e-4:
        ax.imshow(
            np.zeros_like(diff_abs),
            cmap=ONYX_GOLD,
            aspect="equal",
            vmin=0,
            vmax=1,
            interpolation="bilinear",
        )
        ax.text(
            0.5, 0.5,
            "identical\nspecimens",
            transform=ax.transAxes,
            ha="center", va="center",
            color=gold,
            fontsize=15,
            family="serif",
            fontstyle="italic",
        )
    else:
        ax.imshow(
            diff_abs,
            cmap=ONYX_GOLD,
            aspect="equal",
            vmin=0,
            vmax=max(diff_abs.max(), 1e-6),
            interpolation="bilinear",
        )
    ax.set_title(
        f"DIVERGENCE   ·   L2 {l2:.4f}",
        color=gold,
        fontsize=8.5,
        family="serif",
        fontstyle="italic",
        pad=14,
        loc="left",
    )

    fig.patch.set_facecolor(bg)
    fig.tight_layout(pad=2.5)
    return fig


def _read_image(uploaded_file) -> Image.Image:
    return Image.open(io.BytesIO(uploaded_file.getvalue()))


def _verdict_block(score: float, threshold: float) -> None:
    passed = score >= threshold
    word = "Pass." if passed else "Fail."
    klass = "pass" if passed else "fail"
    rel_glyph = "≥" if passed else "<"
    rel_word = "exceeds" if passed else "falls below"

    st.markdown(
        f"""
        <div class="verdict">
          <div class="verdict-eyebrow">— Result of Verification</div>
          <div class="verdict-grid">
            <h2 class="verdict-word {klass}">{word}</h2>
            <div class="verdict-aside">
              <span class="num">{score:.4f}</span><span class="lbl">cosine similarity</span>
              <br/>
              <span class="num">{score*100:.2f}%</span><span class="lbl">confidence</span>
              <br/>
              <span class="num">{rel_glyph} {threshold:.2f}</span><span class="lbl">{rel_word} threshold</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    device = detect_device()
    model, processor = _get_model_and_processor(device)
    param_count = sum(p.numel() for p in model.parameters())

    # ===== MASTHEAD =====
    st.markdown(
        f"""
        <div class="masthead">
          <div class="brand"><span class="mark">◈</span> Emerald-Gold-Ring-Classification</div>
          <div class="meta">
            <span>Edition I · 2026</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ===== HERO =====
    hero_left, hero_right = st.columns([1.55, 1], gap="large")
    with hero_left:
        st.markdown(
            """
            <div class="hero">

              <h1 class="hero-title">
                The verification<br/>
                of <span class="accent">form</span><span class="dot">.</span>
              </h1>
              <p class="hero-sub">
                A reference jewel is held as ground truth. Each new specimen
                is read by a self-supervised vision transformer and judged
                against it — offline, on-device, in milliseconds.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_right:
        st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)
        if "saved_threshold" not in st.session_state:
            st.session_state.saved_threshold = 0.85
        threshold = st.slider(
            "Pass Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.saved_threshold,
            step=0.01,
        )
        if st.button("Save Threshold", use_container_width=True, key="save_thresh"):
            st.session_state.saved_threshold = threshold
            st.success(f"Threshold saved — {threshold:.2f}")
        st.markdown(
            f"""
            <div class="info-panel">
              <div class="info-panel-title">Instrument</div>
              <div class="info-row">
                <span class="info-key">Architecture</span>
                <span class="info-val">DinoV2 · ViT-B/14</span>
              </div>
              <div class="info-row">
                <span class="info-key">Parameters</span>
                <span class="info-val">{param_count/1e6:.1f}M</span>
              </div>
              <div class="info-row">
                <span class="info-key">Embedding</span>
                <span class="info-val">{EMBEDDING_DIM}-d</span>
              </div>
              <div class="info-row">
                <span class="info-key">Metric</span>
                <span class="info-val">cosine</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ===== UPLOAD =====
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_tmpl, col_input = st.columns(2, gap="large")
    with col_tmpl:
        st.markdown(
            '<div class="section-label"><span class="num">i.</span> Reference Template <span class="line"></span></div>',
            unsafe_allow_html=True,
        )
        template_file = st.file_uploader(
            "Template",
            type=["jpg", "jpeg", "png"],
            key="template",
            label_visibility="collapsed",
        )
        template_img = _read_image(template_file) if template_file else None
        if template_img:
            st.image(template_img, use_container_width=True)

    with col_input:
        st.markdown(
            '<div class="section-label"><span class="num">ii.</span> Specimen Input <span class="line"></span></div>',
            unsafe_allow_html=True,
        )
        input_file = st.file_uploader(
            "Input",
            type=["jpg", "jpeg", "png"],
            key="input",
            label_visibility="collapsed",
        )
        input_img = _read_image(input_file) if input_file else None
        if input_img:
            st.image(input_img, use_container_width=True)

    if template_img is None or input_img is None:
        st.markdown(
            '<div class="stage">awaiting both specimens</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
    run_clicked = st.button("Run Verification", use_container_width=True)

    if run_clicked:
        with st.spinner("◈  computing embeddings..."):
            st.session_state.match_result = match(
                template_img, input_img, model, processor, device, threshold
            )

    result = st.session_state.get("match_result")
    if result is None:
        return

    # ===== VERDICT =====
    _verdict_block(result.score, threshold)

    # ===== TIMING =====
    st.markdown(
        '<div class="section-label"><span class="num">iii.</span> Inference Timing <span class="line"></span></div>',
        unsafe_allow_html=True,
    )
    timing_cols = st.columns(4, gap="large")
    timings = [
        ("Preprocess", result.preprocess_ms, False),
        ("Model Forward", result.inference_ms, False),
        ("Similarity", result.similarity_ms, False),
        ("Total", result.total_ms, True),
    ]
    for col, (label, val, gold) in zip(timing_cols, timings):
        gold_class = " gold" if gold else ""
        col.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value{gold_class}">{val:.1f}<span class="unit">ms</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ===== EMBEDDINGS =====
    st.markdown('<div style="height: 3.5rem;"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label"><span class="num">iv.</span> Feature Embeddings <span class="line"></span></div>',
        unsafe_allow_html=True,
    )
    fig = _embedding_figure(result.template_embedding, result.input_embedding)
    st.pyplot(fig, use_container_width=True)

    st.markdown(
        '<div class="annotation">'
        'Each cell is one of 768 learned feature dimensions from DinoV2&#39;s CLS token, '
        'arranged into a 24&times;32 grid for display. These are abstract concepts &mdash; '
        'shape, texture, symmetry &mdash; not pixels, so the grain is expected. The signal '
        'lives in the <em>pattern</em>: two visually similar jewels produce near-identical '
        'imprints. The <em>Divergence</em> panel shows magnitude of per-dimension difference '
        '&mdash; a dark panel means the two embeddings agree everywhere; bright cells expose '
        'precisely where the representations disagree.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 3rem 0 1.5rem 0;
            border-top: 1px solid rgba(244,241,234,0.08);
            margin-top: 4rem;
            font-family: 'Instrument Sans', sans-serif;
            font-size: 0.62rem;
            letter-spacing: 0.32em;
            text-transform: uppercase;
            color: #6B635A;
        ">
            © 2026 All Rights Reserved &nbsp;·&nbsp;
            Made by <span style="color:#C9A961; font-weight:600;">iQube</span>
            <span style="color:#B23C4B;">&#10084;</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
