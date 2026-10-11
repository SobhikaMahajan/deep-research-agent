"""
Application bootstrap and query processing.
Main entry point for the Deep Research Agent.
"""

import os
import sys
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ResearchApplication:
    """Main application class."""

    def __init__(self, app_config=None):
        from config import config as default_config
        self.config = app_config or default_config
        self._initialized = False
        self.orchestrator = None

    def initialize(self):
        """Initialize all services and the orchestrator."""
        if self._initialized:
            return

        logger.info("Initializing ResearchApplication...")

        # 1. Validate configuration
        validation = self.config.validate_all()
        logger.info(f"Config validation: {validation}")

        if not validation.get("groq") and not validation.get("gemini"):
            raise RuntimeError(
                "No LLM API key configured. "
                "Set GROQ_API_KEY and/or GEMINI_API_KEY in your .env file. "
                "Get free keys at https://console.groq.com or https://aistudio.google.com/apikey"
            )

        # 2. Create LLM config for AG2
        llm_config = self._create_llm_config()

        # 3. Create ChromaDB search service
        from services.chroma_search_service import ChromaSearchService
        chroma_service = ChromaSearchService()

        # 4. Load sample documents if collection is empty
        if chroma_service.document_count == 0:
            logger.info("No documents in ChromaDB — loading sample documents...")
            from services.document_loader import create_sample_documents, load_documents_into_chroma
            create_sample_documents()
            load_documents_into_chroma(chroma_service=chroma_service)

        # 5. Create Tavily service
        from services.tavily_service import TavilyService
        tavily_service = TavilyService()

        # 6. Create orchestrator
        from orchestration.group_chat_manager import ResearchGroupChatOrchestrator
        self.orchestrator = ResearchGroupChatOrchestrator(
            llm_config=llm_config,
            chroma_search_service=chroma_service,
            tavily_service=tavily_service,
        )

        self._initialized = True
        logger.info("ResearchApplication initialized successfully")

    def _create_llm_config(self) -> dict:
        """Create AG2-compatible LLM config with Groq primary + Gemini fallback.

        Groq is primary because its free tier allows 30 RPM,
        while Gemini free tier only allows 5 RPM — too low for
        a multi-agent system that makes 10+ calls per query.
        AG2 automatically tries the next provider when one fails.
        """
        config_list = []

        # Primary: Groq (30 RPM free tier)
        groq = self.config.create_fallback_llm_config()
        if groq and groq.validate():
            config_list.append({
                "model": groq.model,
                "api_key": groq.api_key,
                "base_url": groq.base_url,
                "api_type": "openai",
            })
            logger.info(f"LLM primary: Groq model={groq.model}")

        # Fallback: Gemini (5 RPM free tier)
        gemini = self.config.create_llm_config()
        if gemini.validate():
            config_list.append({
                "model": gemini.model,
                "api_key": gemini.api_key,
                "base_url": gemini.base_url,
                "api_type": "openai",
            })
            logger.info(f"LLM fallback: Gemini model={gemini.model}")

        if not config_list:
            raise RuntimeError("No valid LLM configuration found")

        temperature = groq.temperature if groq else gemini.temperature
        return {"config_list": config_list, "temperature": temperature}

    def process_query(self, user_query: str, session_id: str = None):
        """
        Process a user research query.

        Args:
            user_query: The research question
            session_id: Optional session identifier

        Returns:
            The response from the orchestrator
        """
        # Auto-initialize if needed
        if not self._initialized:
            self.initialize()

        from streamlit_callback import capture_agent_messages
        from queue_handler import message_queue

        logger.info(f"Processing query (session={session_id}): {user_query[:100]}...")

        try:
            with capture_agent_messages():
                result = self.orchestrator.process_query(user_query)

            # Signal end of stream
            message_queue.signal_end()

            return result

        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            message_queue.put_error(str(e))
            message_queue.signal_end()
            raise


# Global singleton
_app = None


def get_app() -> ResearchApplication:
    """Get or create the global application instance."""
    global _app
    if _app is None:
        _app = ResearchApplication()
    return _app


# CLI support
def main():
    """Run a query from the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python main.py \"your research question here\"")
        print("\nExample:")
        print('  python main.py "Compare Philips MX800 vs Mindray N22 patient monitors"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"\n🔬 Deep Research Agent")
    print(f"{'=' * 60}")
    print(f"Query: {query}")
    print(f"{'=' * 60}\n")

    app = get_app()
    app.initialize()

    # For CLI, don't use the streaming callback
    result = app.orchestrator.process_query(query)

    print(f"\n{'=' * 60}")
    print("✅ Research complete!")

    # Extract final answer from chat history
    if result and hasattr(result, "chat_history"):
        for msg in reversed(result.chat_history):
            if isinstance(msg, dict) and msg.get("name") == "Research_Manager_Agent":
                print(f"\n📋 Final Answer:\n")
                print(msg.get("content", "No content"))
                break


if __name__ == "__main__":
    main()
