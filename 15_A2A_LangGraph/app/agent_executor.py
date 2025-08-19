import logging
import os

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError

from app.agent import Agent
from app.rag import _get_rag_graph


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneralAgentExecutor(AgentExecutor):
    """General Purpose AgentExecutor with A2A Protocol Support."""

    def __init__(self):
        self.agent = Agent()
        
        # Initialize and verify RAG system
        self._initialize_rag_system()

    def _initialize_rag_system(self):
        """Initialize and verify the RAG system is properly configured."""
        try:
            # Get RAG data directory from environment
            rag_data_dir = os.getenv('RAG_DATA_DIR', 'data')
            logger.info(f"RAG system configured to use data directory: {rag_data_dir}")
            
            # Check if data directory exists
            if os.path.exists(rag_data_dir):
                pdf_files = [f for f in os.listdir(rag_data_dir) if f.endswith('.pdf')]
                logger.info(f"Found {len(pdf_files)} PDF files in RAG data directory: {pdf_files}")
            else:
                logger.warning(f"RAG data directory does not exist: {rag_data_dir}")
            
            # Initialize the RAG graph to verify it works
            rag_graph = _get_rag_graph()
            logger.info("✅ RAG system initialized successfully")
            
            # Test RAG retrieval (optional verification)
            try:
                test_result = rag_graph.invoke({"question": "test"})
                if test_result:
                    logger.info("✅ RAG system test query successful")
                else:
                    logger.warning("⚠️ RAG system test query returned empty result")
            except Exception as test_error:
                logger.warning(f"⚠️ RAG system test query failed: {test_error}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG system: {e}")
            logger.error("RAG functionality may not work properly")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        try:
            logger.info(f"Starting agent stream for query: {query}")
            
            # Log available tools for debugging
            from app.tools import get_tool_belt
            tools = get_tool_belt()
            tool_names = [tool.name if hasattr(tool, 'name') else str(type(tool).__name__) for tool in tools]
            logger.info(f"Agent has access to tools: {tool_names}")
            
            # Check if query might benefit from RAG
            if any(keyword in query.lower() for keyword in ['document', 'policy', 'student loan', 'evaluation', 'rag']):
                logger.info("Query appears to be RAG-related - document retrieval tool should be useful")
            
            async for item in self.agent.stream(query, task.context_id):
                is_task_complete = item['is_task_complete']
                require_user_input = item['require_user_input']
                logger.info(f"Stream item - complete: {is_task_complete}, requires_input: {require_user_input}")

                if not is_task_complete and not require_user_input:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            item['content'],
                            task.context_id,
                            task.id,
                        ),
                    )
                elif require_user_input:
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            item['content'],
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                    break
                else:
                    # Log the final response content for debugging
                    content = item['content']
                    logger.info(f"Final response content (first 200 chars): {content[:200]}...")
                    
                    await updater.add_artifact(
                        [Part(root=TextPart(text=content))],
                        name='result',
                    )
                    await updater.complete()
                    break

        except Exception as e:
            logger.error(f'An error occurred while streaming the response: {e}')
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())
