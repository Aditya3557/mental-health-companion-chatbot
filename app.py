import os
import time
from datetime import datetime
from typing import List, Dict

import streamlit as st
import pandas as pd
import altair as alt

# Gemini (Google Generative AI)
try:
    import google.generativeai as genai
except Exception as e:
    genai = None

APP_TITLE = "Student Mental Health Companion"
APP_ICON = "💬"
MODEL_NAME_DEFAULT = "gemini-2.0-flash"

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# Global minimal CSS for pinned input and cards
st.markdown(
    """
    <style>
    /* Keep chat input visually at bottom across layouts */
    [data-testid=\"stChatInput\"] { position: sticky; bottom: 0; z-index: 999; }
    /* Card styles */
    .mood-card { padding: 1rem; border-radius: 12px; border: 1px solid #dbeafe; background: linear-gradient(180deg,#eff6ff, #ffffff); color:#000; }
    .mood-card .label { color:#000; font-weight:600; }
    .mood-card .value { font-size: 1.75rem; font-weight: 700; color:#000; }
    .mood-card .sub { font-size: 0.85rem; color:#000; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Utilities and configuration
# ---------------------------

def get_api_key_from_env_or_secrets() -> str:
    # Prefer Streamlit secrets if available; fallback to env var
    key = None
    try:
        key = st.secrets.get("GOOGLE_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        pass
    return key or os.getenv("GOOGLE_API_KEY", "")


def config_gemini(api_key: str):
    if genai is None:
        st.error("google-generativeai is not installed. Please run: pip install google-generativeai")
        st.stop()
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to configure Gemini: {e}")
        st.stop()


def list_available_chat_models() -> List[str]:
    """Return available model names that support text generation for this key.
    Falls back to empty list on error. Names are normalized without the 'models/' prefix.
    """
    if genai is None:
        return []
    try:
        models = genai.list_models()
    except Exception:
        return []

    names: List[str] = []
    for m in models:
        name = getattr(m, "name", None)
        if not name:
            continue
        methods = getattr(m, "supported_generation_methods", None)
        if methods is None:
            methods = getattr(m, "generation_methods", None)
        try:
            method_set = set([str(x) for x in methods]) if methods else set()
        except Exception:
            method_set = set()
        # Include models that can generate text content
        if ("generateContent" in method_set) or ("generate_text" in method_set) or ("create" in method_set):
            short = name.split("/")[-1]
            names.append(short)

    # Deduplicate preserving order; prefer common names first
    preferred = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro", "gemini-pro"]
    ordered: List[str] = []
    for p in preferred:
        if p in names and p not in ordered:
            ordered.append(p)
    for n in names:
        if n not in ordered:
            ordered.append(n)
    return ordered


def mood_face(score: int) -> str:
    if score >= 8:
        return "😄"
    if score >= 6:
        return "🙂"
    if score >= 4:
        return "😐"
    if score >= 2:
        return "🙁"
    return "😢"


def mood_text(score: int) -> str:
    if score >= 8:
        return "Very positive"
    if score >= 6:
        return "Positive"
    if score >= 4:
        return "Neutral"
    if score >= 2:
        return "Low"
    return "Very low"


def build_model(model_name: str):
    system_instruction = (
        "You are a supportive, professional mental health companion for students. "
        "Offer empathetic, evidence-informed guidance for stress, anxiety, academic pressure, time management, and self-care. "
        "You are not a clinician and do not diagnose. Encourage seeking professional help when concerns persist. "
        "If there are indications of crisis (self-harm, intent to harm, severe distress), you must advise immediate help and crisis resources. "
        "Keep responses practical, concise, and actionable. Suggest short exercises (breathing, grounding, scheduling, reframing) when helpful."
    )

    try:
        model = genai.GenerativeModel(
            model_name,
            system_instruction=system_instruction,
        )
        return model
    except Exception as e:
        st.error(f"Failed to create model: {e}")
        st.stop()


# ---------------------------
# Session state
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = [
        {
            "role": "assistant",
            "content": (
                "Hi there! I'm here to support you. How are you feeling today? "
                "You can tell me about stress, anxiety, study pressure, sleep, or anything on your mind."
            ),
        }
    ]

if "mood_history" not in st.session_state:
    st.session_state.mood_history: List[Dict[str, float]] = []

if "chat" not in st.session_state:
    st.session_state.chat = None
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "latencies" not in st.session_state:
    st.session_state.latencies: List[float] = []


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")

    st.markdown("""
    This app is not a substitute for professional care. If you're in immediate danger or thinking about harming yourself, seek help right away:
    - India : Call or text 14416 (Suicide & Crisis Lifeline)
    - International: Visit https://findahelpline.com or contact local emergency services
    """)

    # Current mood in sidebar
    latest_mood_sb = st.session_state.mood_history[-1]["mood"] if st.session_state.mood_history else None
    if latest_mood_sb is not None:
        prev_sb = st.session_state.mood_history[-2]["mood"] if len(st.session_state.mood_history) > 1 else None
        delta_sb = None if prev_sb is None else latest_mood_sb - prev_sb
        st.markdown(
            f"""
            <div class='mood-card'>
              <div class='label'>Current mood</div>
              <div class='value'>{int(latest_mood_sb)}/10 {mood_face(int(latest_mood_sb))}</div>
              <div class='sub'>{mood_text(int(latest_mood_sb))}{' (Δ ' + ('+' if delta_sb and delta_sb>0 else '') + str(delta_sb) + ')' if delta_sb is not None else ''}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("No mood saved yet — use the Mood check-in below.")
    # Apply API key and lock to Flash model (no choices)
    api_key_final = get_api_key_from_env_or_secrets()
    target_model = MODEL_NAME_DEFAULT  # gemini-1.5-flash
    if not api_key_final:
        st.warning("Provide your Google API key via environment variable or Streamlit secrets. See README for setup.")
    else:
        config_gemini(api_key_final)
        # Start chat if needed
        if st.session_state.chat is None or st.session_state.current_model is None:
            # Try Flash first
            try:
                st.session_state.chat = build_model(target_model).start_chat(history=[])
                st.session_state.current_model = target_model
            except Exception as e:
                # Fallback to a widely available model without exposing stack trace details
                fallback = "gemini-pro"
                try:
                    st.session_state.chat = build_model(fallback).start_chat(history=[])
                    st.session_state.current_model = fallback
                    st.info("gemini-1.5-flash is not available for your API/version. Falling back to gemini-pro.")
                except Exception as e2:
                    st.error(f"Failed to start chat with Flash or fallback: {e2}")
        st.caption(f"Using model: {st.session_state.current_model or target_model}")

    st.divider()
    st.caption("Privacy: Conversations are kept only in this session unless you explicitly export them.")


# ---------------------------
# Quick prompts / helpers
# ---------------------------
col1, col2, col3 = st.columns(3)
with col1:
    qp1 = st.button("I'm overwhelmed with assignments")
with col2:
    qp2 = st.button("Test anxiety before exams")
with col3:
    qp3 = st.button("Trouble sleeping and focusing")

quick_prompt = None
if qp1:
    quick_prompt = "I'm overwhelmed with assignments; help me plan and reduce stress."
elif qp2:
    quick_prompt = "I have test anxiety. Please suggest practical techniques to calm down and study effectively."
elif qp3:
    quick_prompt = "I'm having trouble sleeping and focusing. Suggest sleep hygiene and focus tips."


# ---------------------------
# Session insights & graphs
# ---------------------------
with st.expander("Session insights & graphs"):
    cols = st.columns(2)

    # Messages by role
    if st.session_state.messages:
        role_counts = (
            pd.DataFrame(st.session_state.messages)
            .groupby("role")
            .size()
            .reset_index(name="count")
        )
        chart_roles = (
            alt.Chart(role_counts)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(x=alt.X("count:Q", title="Messages"), y=alt.Y("role:N", title="Role"))
            .properties(height=120)
        )
        with cols[0]:
            st.altair_chart(chart_roles, use_container_width=True)

    # Response latency per turn
    if st.session_state.latencies:
        df_lat = pd.DataFrame({
            "Turn": list(range(1, len(st.session_state.latencies) + 1)),
            "Latency_ms": [round(ms, 0) for ms in st.session_state.latencies],
        })
        chart_lat = (
            alt.Chart(df_lat)
            .mark_line(point=True)
            .encode(x="Turn:Q", y=alt.Y("Latency_ms:Q", title="Model response time (ms)"))
            .properties(height=160)
        )
        with cols[1]:
            st.altair_chart(chart_lat, use_container_width=True)

    st.divider()


# ---------------------------
# Mood check-in
# ---------------------------
with st.expander("Mood check-in (session only)"):
    cols_m = st.columns([2,1])
    with cols_m[0]:
        mood = st.slider("How are you feeling right now?", 0, 10, 5)
        if st.button("Save mood"):
            st.session_state.mood_history.append({"t": time.time(), "mood": mood})
    with cols_m[1]:
        st.caption("Your latest mood appears in the sidebar.")


# ---------------------------
# Crisis detection helper
# ---------------------------
CRISIS_KEYWORDS = [
    "suicide", "hurt myself", "kill myself", "self-harm", "end my life", "no reason to live",
    "harm others", "kill someone", "can't go on", "despair", "hopeless"
]


def maybe_show_crisis_banner(text: str):
    lowered = text.lower()
    if any(kw in lowered for kw in CRISIS_KEYWORDS):
        st.error(
            "If you are in crisis or considering harming yourself or others, please seek immediate help: "
            "US: Call or text 988. International: https://findahelpline.com or your local emergency services."
        )


# ---------------------------
# Chat UI
# ---------------------------
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


# ---------------------------
# Chat input (pinned at bottom)
# ---------------------------
user_input = quick_prompt if quick_prompt else None
user_input = st.chat_input("Type your message…") or user_input

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    maybe_show_crisis_banner(user_input)

    # Get model response with latency timing
    if st.session_state.chat is None:
        st.session_state.messages.append({"role": "assistant", "content": "Please provide a valid API key in the sidebar to start the chat."})
    else:
        try:
            t0 = time.time()
            response = st.session_state.chat.send_message(user_input)
            t1 = time.time()
            elapsed_ms = (t1 - t0) * 1000.0
            st.session_state.latencies.append(elapsed_ms)
            assistant_text = response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            assistant_text = f"I had trouble contacting the model: {e}"
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    st.rerun()

# ---------------------------
# Footer
# ---------------------------
st.divider()
col_a, col_b = st.columns([2, 1])
with col_a:
    st.caption(
        "This tool provides supportive information only and does not replace professional advice. "
        "If you're struggling, consider reaching out to a counselor, therapist, or a trusted person."
    )
with col_b:
    st.caption("Powered by Google Gemini")
