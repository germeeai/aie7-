"""Production-safe LangGraph agent with integrated guardrails."""

import logging
from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from .models import get_openai_model
from .rag import ProductionRAGChain
from .agents import get_default_tools
from .guardrails_nodes import (
    GuardrailsValidator,
    GuardrailsState,
    create_input_validation_node,
    create_output_validation_node,
    create_refinement_node,
    should_validate_input,
    should_validate_output,
    should_refine
)


logger = logging.getLogger(__name__)


class SafeAgentConfig:
    """Configuration for the production-safe agent."""
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.1,
        allowed_topics: List[str] = None,
        competitors: List[str] = None,
        pii_entities: List[str] = None,
        max_refinements: int = 3,
        enable_logging: bool = True
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.allowed_topics = allowed_topics or [
            "student loans", "financial aid", "education funding", 
            "FAFSA", "scholarships", "grants", "loan repayment"
        ]
        self.competitors = competitors or [
            "OpenAI", "Anthropic", "Google", "Microsoft", "Meta",
            "ChatGPT", "Claude", "Bard", "Copilot"
        ]
        self.pii_entities = pii_entities or [
            "CREDIT_CARD", "SSN", "PHONE_NUMBER", "EMAIL_ADDRESS",
            "PERSON", "DATE_TIME", "LOCATION"
        ]
        self.max_refinements = max_refinements
        self.enable_logging = enable_logging


class ProductionSafeAgent:
    """Production-ready LangGraph agent with comprehensive guardrails."""
    
    def __init__(
        self,
        config: SafeAgentConfig,
        rag_chain: Optional[ProductionRAGChain] = None,
        tools: Optional[List] = None
    ):
        """Initialize the production-safe agent.
        
        Args:
            config: Agent configuration
            rag_chain: Optional RAG chain for document retrieval
            tools: Optional list of tools for the agent
        """
        self.config = config
        self.rag_chain = rag_chain
        
        # Setup logging
        if config.enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Initialize guardrails validator
        self.validator = GuardrailsValidator(
            allowed_topics=config.allowed_topics,
            competitors=config.competitors,
            pii_entities=config.pii_entities,
            max_refinements=config.max_refinements
        )
        
        # Setup tools
        self.tools = tools or get_default_tools(rag_chain)
        
        # Initialize model
        self.model = get_openai_model(
            model_name=config.model_name,
            temperature=config.temperature
        )
        self.model_with_tools = self.model.bind_tools(self.tools)
        
        # Build the graph
        self.graph = self._build_graph()
        logger.info("✓ Production-safe agent initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph with integrated guardrails."""
        
        # Create graph with extended state
        graph = StateGraph(GuardrailsState)
        
        # Create nodes
        input_validator = create_input_validation_node(self.validator)
        output_validator = create_output_validation_node(self.validator)
        refinement_node = create_refinement_node(self.validator)
        tool_node = ToolNode(self.tools)
        
        # Add nodes to graph
        graph.add_node("input_validation", input_validator)
        graph.add_node("agent", self._call_model)
        graph.add_node("output_validation", output_validator)
        graph.add_node("tools", tool_node)
        graph.add_node("refinement", refinement_node)
        graph.add_node("blocked", self._handle_blocked_input)
        graph.add_node("error", self._handle_error)
        
        # Set entry point
        graph.set_entry_point("input_validation")
        
        # Add conditional edges for input validation
        graph.add_conditional_edges(
            "input_validation",
            should_validate_input,
            {
                "continue": "agent",
                "blocked": "blocked",
                "error": "error"
            }
        )
        
        # Add conditional edges for agent decisions
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "validate_output": "output_validation"
            }
        )
        
        # Tools always go back to agent
        graph.add_edge("tools", "agent")
        
        # Add conditional edges for output validation
        graph.add_conditional_edges(
            "output_validation",
            should_validate_output,
            {
                "complete": END,
                "refine": "refinement"
            }
        )
        
        # Add conditional edges for refinement
        graph.add_conditional_edges(
            "refinement",
            should_refine,
            {
                "retry": "agent",
                "complete": END
            }
        )
        
        # End states
        graph.add_edge("blocked", END)
        graph.add_edge("error", END)
        
        return graph.compile()
    
    def _call_model(self, state: GuardrailsState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        
        # Add system message if not present
        if not messages or not any(msg.type == "system" for msg in messages if hasattr(msg, 'type')):
            system_message = HumanMessage(
                content="You are a helpful assistant specializing in student loans and financial aid. "
                       "Provide accurate, helpful information while staying on topic. "
                       "Do not mention competitor companies or include any personal information."
            )
            messages = [system_message] + messages
        
        try:
            response = self.model_with_tools.invoke(messages)
            logger.info("✓ Model response generated")
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Model invocation failed: {e}")
            error_message = AIMessage(
                content="I apologize, but I'm experiencing technical difficulties. Please try again."
            )
            return {"messages": [error_message]}
    
    def _should_continue(self, state: GuardrailsState) -> str:
        """Route decision for agent actions."""
        messages = state["messages"]
        if not messages:
            return "validate_output"
        
        last_message = messages[-1]
        
        # Check if the message has tool calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info("Tool calls detected, routing to tools")
            return "tools"
        else:
            logger.info("No tool calls, routing to output validation")
            return "validate_output"
    
    def _handle_blocked_input(self, state: GuardrailsState) -> Dict[str, Any]:
        """Handle blocked input scenarios."""
        failures = state.get("guard_failures", [])
        logger.warning(f"Input blocked due to: {failures}")
        
        # Create appropriate response based on failure type
        if any("jailbreak" in failure.lower() for failure in failures):
            response = "I cannot respond to requests that attempt to bypass safety guidelines."
        elif any("topic" in failure.lower() for failure in failures):
            response = ("I can only help with questions about student loans, financial aid, and education funding. "
                       "Please ask a question related to these topics.")
        elif any("pii" in failure.lower() for failure in failures):
            response = "Please don't include personal information in your message for your privacy and security."
        else:
            response = "I cannot process your request. Please rephrase your question."
        
        blocked_message = AIMessage(content=response)
        return {
            "messages": [blocked_message],
            "validation_results": {"blocked": True, "reason": failures}
        }
    
    def _handle_error(self, state: GuardrailsState) -> Dict[str, Any]:
        """Handle system errors gracefully."""
        logger.error("System error encountered in guardrails validation")
        
        error_message = AIMessage(
            content="I'm experiencing technical difficulties with safety validation. "
                   "Please try your request again, or contact support if the issue persists."
        )
        return {
            "messages": [error_message],
            "validation_results": {"error": True}
        }
    
    def invoke(self, message: str) -> Dict[str, Any]:
        """Invoke the safe agent with a user message.
        
        Args:
            message: User input message
            
        Returns:
            Agent response with validation metadata
        """
        logger.info(f"Processing user message: {message[:100]}...")
        
        # Create initial state
        initial_state = GuardrailsState(
            messages=[HumanMessage(content=message)],
            validation_results={},
            guard_failures=[],
            refinement_count=0,
            max_refinements=self.config.max_refinements
        )
        
        try:
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            # Extract the final response
            final_messages = result.get("messages", [])
            final_response = final_messages[-1].content if final_messages else "No response generated."
            
            # Compile response with metadata
            response = {
                "response": final_response,
                "validation_results": result.get("validation_results", {}),
                "guard_failures": result.get("guard_failures", []),
                "refinement_count": result.get("refinement_count", 0),
                "safe": len(result.get("guard_failures", [])) == 0
            }
            
            logger.info(f"Agent response completed. Safe: {response['safe']}")
            return response
            
        except Exception as e:
            logger.error(f"Agent invocation failed: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your request.",
                "validation_results": {"error": str(e)},
                "guard_failures": ["system_error"],
                "refinement_count": 0,
                "safe": False
            }
    
    async def ainvoke(self, message: str) -> Dict[str, Any]:
        """Async version of invoke method."""
        # For now, delegate to sync version
        # In production, you'd implement full async support
        return self.invoke(message)
    
    def get_conversation_history(self, state: GuardrailsState) -> List[str]:
        """Extract conversation history from state."""
        history = []
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage):
                history.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                history.append(f"Assistant: {msg.content}")
        return history


def create_safe_agent(
    config: SafeAgentConfig = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    tools: Optional[List] = None
) -> ProductionSafeAgent:
    """Factory function to create a production-safe agent.
    
    Args:
        config: Agent configuration (uses defaults if None)
        rag_chain: Optional RAG chain for document retrieval
        tools: Optional tools for the agent
        
    Returns:
        Configured ProductionSafeAgent instance
    """
    if config is None:
        config = SafeAgentConfig()
    
    return ProductionSafeAgent(config, rag_chain, tools)