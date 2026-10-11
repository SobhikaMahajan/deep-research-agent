"""
Streamlit frontend for the Deep Research Agent.
Chat UI with real-time agent message streaming.
"""

import time
import queue
import threading
import logging

import streamlit as st

from main import get_app
from queue_handler import message_queue
from streamlit_helpers import (
    CUSTOM_CSS,
    initialize_session,
    sample_questions_sidebar,
    display_chat_history,
    display_error,
    render_spinner,
)

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI-Powered Research — Deep Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def process_query_async(prompt: str, session_id: str):
    """Run query processing in a background thread."""
    try:
        app = get_app()
        app.process_query(prompt, session_id=session_id)
    except Exception as e:
        logger.error(f"Background query processing failed: {e}", exc_info=True)
        message_queue.put_error(str(e))
        message_queue.signal_end()


def render_response_stream(
    reasoning_placeholder,
    final_answer_placeholder,
    reasoning_expander,
    timeout: int = 120,
):
    """
    Poll the message queue and stream responses to the UI.

    Args:
        reasoning_placeholder: Streamlit placeholder for reasoning content
        final_answer_placeholder: Streamlit placeholder for final answer
        reasoning_expander: Streamlit expander for reasoning trace
        timeout: Maximum seconds to wait for responses

    Returns:
        Tuple of (final_answer_content, reasoning_content)
    """
    reasoning_content = ""
    final_answer_content = ""
    start_time = time.time()

    while True:
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout:
            display_error(f"Response timed out after {timeout}s")
            break

        try:
            msg = message_queue.get(block=True, timeout=2.0)

            # None = end of stream
            if msg is None:
                break

            agent_name = msg.agent_name or ""
            content = msg.content or ""

            if not content:
                continue

            # Route messages based on agent
            if agent_name == "Research_Manager_Agent":
                # Final answer
                final_answer_content += content + "\n"
                final_answer_placeholder.markdown(
                    f'<div class="final-answer">{final_answer_content}</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Reasoning / intermediate messages
                import html as html_lib
                escaped = html_lib.escape(content)
                reasoning_content += f"\n**[{agent_name}]**\n{escaped}\n\n---\n"
                with reasoning_expander:
                    reasoning_placeholder.markdown(
                        f'<div class="reasoning-box">{reasoning_content}</div>',
                        unsafe_allow_html=True,
                    )

        except queue.Empty:
            # No message yet, continue polling
            continue
        except Exception as e:
            logger.error(f"Error reading from queue: {e}")
            break

    return final_answer_content, reasoning_content


def process_prompt(prompt: str):
    """Process a user prompt: stream agents, display results."""
    # 1. Append user message to history
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # 2. Clear the message queue
    message_queue.clear()

    # 3. Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. Start background processing thread
    session_id = st.session_state.get("session_id", "default")
    thread = threading.Thread(
        target=process_query_async,
        args=(prompt, session_id),
        daemon=True,
    )

    start_time = time.time()

    # 5. Create UI placeholders
    with st.chat_message("assistant"):
        # Spinner
        spinner_placeholder = st.empty()
        spinner_placeholder.markdown(render_spinner("Researching..."), unsafe_allow_html=True)

        # Reasoning expander
        reasoning_expander = st.expander("🧠 Agent Reasoning Trace", expanded=True)
        with reasoning_expander:
            reasoning_placeholder = st.empty()

        # Final answer placeholder
        st.markdown("### 📋 Research Results")
        final_answer_placeholder = st.empty()

        # Start the thread
        thread.start()

        # 6. Stream responses
        final_answer, reasoning = render_response_stream(
            reasoning_placeholder=reasoning_placeholder,
            final_answer_placeholder=final_answer_placeholder,
            reasoning_expander=reasoning_expander,
        )

        # 7. Remove spinner
        spinner_placeholder.empty()

        # 8. Show latency
        elapsed = time.time() - start_time
        st.markdown(
            f'<div class="latency-info">'
            f'<span class="icon">⏱️</span> Response time: {elapsed:.1f}s'
            f'</div>',
            unsafe_allow_html=True,
        )

    # 9. Append assistant message to history
    st.session_state["messages"].append({
        "role": "assistant",
        "content": final_answer,
        "reasoning": reasoning,
        "latency": elapsed,
    })


def main():
    """Main Streamlit application."""
    # Initialize session
    initialize_session()

    # Sidebar with sample questions
    sample_questions_sidebar()

    # Title
    st.markdown("# 🔬 Deep Research Agent")
    st.markdown(
        "Multi-agent research system for competitive intelligence. "
        "Ask a question to get comprehensive analysis with citations."
    )
    st.markdown("---")

    # Display chat history
    display_chat_history()

    # Handle sample question selection
    if "selected_question" in st.session_state:
        selected = st.session_state.pop("selected_question")
        process_prompt(selected)
        st.rerun()

    # Handle chat input
    if prompt := st.chat_input("Ask a research question..."):
        process_prompt(prompt)


if __name__ == "__main__":
    main()
else:
    # When run via `streamlit run app.py`
    main()
