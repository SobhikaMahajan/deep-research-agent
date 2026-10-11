"""
Streamlit UI components — CSS, session management, sidebar, and display helpers.
"""

import uuid
import html
import re
import streamlit as st
from dataclasses import dataclass
from typing import List, Optional


# --- Custom CSS ---

CUSTOM_CSS = """
<style>
/* Main container */
.stApp {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* Reasoning box */
.reasoning-box {
    background-color: #1e1e2e;
    border: 1px solid #45475a;
    border-radius: 10px;
    padding: 16px;
    margin: 8px 0;
    max-height: 400px;
    overflow-y: auto;
    font-size: 0.85em;
    line-height: 1.6;
    color: #cdd6f4;
}

.reasoning-box h1, .reasoning-box h2, .reasoning-box h3 {
    color: #89b4fa;
    font-size: 1em;
    margin-top: 12px;
}

.reasoning-box code {
    background-color: #313244;
    padding: 2px 6px;
    border-radius: 4px;
    color: #a6e3a1;
}

/* Final answer */
.final-answer {
    background-color: #0d1117;
    border-left: 4px solid #58a6ff;
    border-radius: 8px;
    padding: 20px 24px;
    margin: 12px 0;
    line-height: 1.8;
    color: #e6edf3;
    font-size: 0.95em;
}

.final-answer h1 { font-size: 1.4em; color: #58a6ff; }
.final-answer h2 { font-size: 1.2em; color: #79c0ff; }
.final-answer h3 { font-size: 1.05em; color: #a5d6ff; }
.final-answer a { color: #58a6ff; text-decoration: none; }
.final-answer a:hover { text-decoration: underline; }

.final-answer table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
}

.final-answer th, .final-answer td {
    border: 1px solid #30363d;
    padding: 8px 12px;
    text-align: left;
}

.final-answer th {
    background-color: #161b22;
    color: #58a6ff;
}

/* Latency info */
.latency-info {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 8px 0;
    font-size: 0.8em;
    color: #8b949e;
    display: flex;
    align-items: center;
    gap: 8px;
}

.latency-info .icon { font-size: 1.1em; }

/* Spinner / Loader */
.loader {
    border: 3px solid #30363d;
    border-top: 3px solid #58a6ff;
    border-radius: 50%;
    width: 20px;
    height: 20px;
    animation: spin 0.8s linear infinite;
    display: inline-block;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.thinking-container {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background-color: #161b22;
    border-radius: 8px;
    margin: 8px 0;
    color: #8b949e;
}

/* Sidebar styling */
.sidebar-question {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
    margin: 6px 0;
    cursor: pointer;
    transition: all 0.2s ease;
}

.sidebar-question:hover {
    border-color: #58a6ff;
    background-color: #1c2333;
}

.category-badge {
    display: inline-block;
    background-color: #1f6feb33;
    color: #58a6ff;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75em;
    margin-bottom: 4px;
}

/* Error display */
.error-box {
    background-color: #2d1b1b;
    border: 1px solid #f85149;
    border-radius: 8px;
    padding: 16px;
    color: #ffa198;
    margin: 8px 0;
}

/* Info display */
.info-box {
    background-color: #1b2d3d;
    border: 1px solid #58a6ff;
    border-radius: 8px;
    padding: 16px;
    color: #a5d6ff;
    margin: 8px 0;
}
</style>
"""


@dataclass
class SampleQuestion:
    """A sample question for the sidebar."""
    text: str
    category: Optional[str] = None


DEFAULT_SAMPLE_QUESTIONS = [
    SampleQuestion(
        text="Help me create a competitive strategy against mindray's latest product launched in 2025 in patient monitoring segment. Focus on strengths and weaknesses comparison.",
        category="Strategy",
    ),
    SampleQuestion(
        text="Compare MX750 VS GE Healthcare products in patient monitoring segment. Include technical specifications and pricing analysis.",
        category="Comparison",
    ),
    SampleQuestion(
        text="Consider philips mx400. Which is the competitor of this product in market? Analyze the competitive landscape.",
        category="Competition",
    ),
]


def render_spinner(message: str = "Thinking...") -> str:
    """Return HTML for a thinking spinner."""
    return f"""
    <div class="thinking-container">
        <div class="loader"></div>
        <span>{html.escape(message)}</span>
    </div>
    """


def sample_questions_sidebar(
    questions: Optional[List[SampleQuestion]] = None,
    show_clear_button: bool = True,
):
    """Render sample questions in the sidebar."""
    questions = questions or DEFAULT_SAMPLE_QUESTIONS

    with st.sidebar:
        st.markdown("### 💡 Sample Questions")
        st.markdown("Click a question to try it:")
        st.markdown("---")

        for i, q in enumerate(questions):
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                if q.category:
                    st.markdown(f'<span class="category-badge">{q.category}</span>', unsafe_allow_html=True)
                # Truncate for display
                display_text = q.text[:120] + "..." if len(q.text) > 120 else q.text
                if st.button(display_text, key=f"sample_q_{i}", use_container_width=True):
                    st.session_state["selected_question"] = q.text

        if show_clear_button:
            st.markdown("---")
            if st.button("🗑️ Clear Chat", use_container_width=True):
                clear_session()
                st.rerun()

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        st.markdown(f"**Model:** Gemini 2.5 Flash")
        st.markdown(f"**Search:** ChromaDB + Tavily")


def clear_session():
    """Clear the session state."""
    st.session_state["messages"] = []
    st.session_state["session_id"] = str(uuid.uuid4())
    if "selected_question" in st.session_state:
        del st.session_state["selected_question"]


def initialize_session():
    """Initialize session state if not already done."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())


def _process_reasoning_content(content: str) -> str:
    """Process reasoning content — escape HTML, reduce heading sizes."""
    escaped = html.escape(content)
    # Reduce heading sizes: # → ###
    escaped = re.sub(r"^(#{1,2})\s", r"### ", escaped, flags=re.MULTILINE)
    return escaped


def display_chat_history():
    """Display all messages in the chat history."""
    for msg in st.session_state.get("messages", []):
        role = msg.get("role", "user")

        if role == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif role == "assistant":
            with st.chat_message("assistant"):
                # Show reasoning in expander if present
                reasoning = msg.get("reasoning", "")
                if reasoning:
                    with st.expander("🧠 Agent Reasoning Trace", expanded=False):
                        processed = _process_reasoning_content(reasoning)
                        st.markdown(
                            f'<div class="reasoning-box">{processed}</div>',
                            unsafe_allow_html=True,
                        )

                # Show final answer
                final = msg.get("content", "")
                if final:
                    st.markdown(
                        f'<div class="final-answer">{final}</div>',
                        unsafe_allow_html=True,
                    )

                # Show latency if present
                latency = msg.get("latency", None)
                if latency:
                    st.markdown(
                        f'<div class="latency-info"><span class="icon">⏱️</span> Response time: {latency:.1f}s</div>',
                        unsafe_allow_html=True,
                    )


def display_error(error_message: str):
    """Display an error message."""
    st.markdown(
        f'<div class="error-box">❌ {html.escape(error_message)}</div>',
        unsafe_allow_html=True,
    )


def display_info(info_message: str):
    """Display an info message."""
    st.markdown(
        f'<div class="info-box">ℹ️ {html.escape(info_message)}</div>',
        unsafe_allow_html=True,
    )
