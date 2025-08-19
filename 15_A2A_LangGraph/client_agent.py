"""
Simple Agent that makes API calls to the A2A Agent through LangGraph.

This creates a LangGraph client agent that can communicate with the A2A server,
demonstrating how one agent can use another agent's capabilities through the A2A protocol.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Annotated, TypedDict
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

# Try to import the newer ClientFactory if available
try:
    from a2a.client import ClientFactory
    USE_CLIENT_FACTORY = True
except ImportError:
    USE_CLIENT_FACTORY = False

# Load environment variables
load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClientAgentState(TypedDict):
    """State schema for the client agent."""
    messages: Annotated[List, add_messages]
    a2a_response: Any
    task_complete: bool


class SimpleA2AClientAgent:
    """Simple agent that makes API calls to the A2A server through LangGraph."""
    
    def __init__(self, a2a_server_url: str = None):
        # Use environment variables with fallback defaults
        self.a2a_server_url = a2a_server_url or os.getenv('A2A_SERVER_URL', 'http://localhost:10000')
        
        # Initialize LLM with environment variables
        self.client_llm = ChatOpenAI(
            model=os.getenv('TOOL_LLM_NAME', 'gpt-4o-mini'),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openai_api_base=os.getenv('TOOL_LLM_URL', 'https://api.openai.com/v1'),
            temperature=0.7
        )
        
        # Validate that we have the required API key
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY environment variable is required but not set")
        
        # Log configuration (without exposing sensitive data)
        logger.info(f"Initializing SimpleA2AClientAgent with:")
        logger.info(f"  - A2A Server URL: {self.a2a_server_url}")
        logger.info(f"  - LLM Model: {os.getenv('TOOL_LLM_NAME', 'gpt-4o-mini')}")
        logger.info(f"  - LLM URL: {os.getenv('TOOL_LLM_URL', 'https://api.openai.com/v1')}")
        logger.info(f"  - OpenAI API Key: {'✓ Set' if os.getenv('OPENAI_API_KEY') else '✗ Missing'}")
        logger.info(f"  - Tavily API Key: {'✓ Set' if os.getenv('TAVILY_API_KEY') else '✗ Missing'}")
        
    async def initialize_a2a_client(self) -> tuple[A2AClient, httpx.AsyncClient]:
        """Initialize the A2A client by fetching the agent card and return both client and httpx_client."""
        httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=self.a2a_server_url
        )
        
        try:
            agent_card = await resolver.get_agent_card()
            logger.info(f"Successfully fetched agent card: {agent_card.name}")
            
            # Use the newer ClientFactory if available, otherwise fall back to A2AClient
            if USE_CLIENT_FACTORY:
                try:
                    client_factory = ClientFactory()
                    a2a_client = await client_factory.create_client(
                        agent_card=agent_card,
                        httpx_client=httpx_client
                    )
                    logger.info("Using ClientFactory for A2A client")
                except Exception as factory_error:
                    logger.warning(f"ClientFactory failed, falling back to A2AClient: {factory_error}")
                    a2a_client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            else:
                a2a_client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
            return a2a_client, httpx_client
        except Exception as e:
            logger.error(f"Failed to initialize A2A client: {e}")
            await httpx_client.aclose()  # Clean up on error
            raise
    
    def build_client_graph(self):
        """Build the LangGraph for the client agent."""
        
        async def analyze_query_node(state: ClientAgentState) -> Dict[str, Any]:
            """Analyze the user query to decide if A2A call is needed."""
            last_message = state["messages"][-1]
            
            analysis_prompt = f"""
            Analyze this user query: "{last_message.content}"
            
            Determine if this query would benefit from using an agent with these capabilities:
            - Web search for current information
            - Academic paper search on arXiv
            - Document retrieval from a knowledge base
            
            Respond with either:
            - "USE_A2A: [reason why A2A agent would be helpful]"
            - "DIRECT: [simple response without needing external tools]"
            """
            
            response = await self.client_llm.ainvoke([HumanMessage(content=analysis_prompt)])
            
            return {
                "messages": [AIMessage(content=f"Analysis: {response.content}")]
            }
        
        async def make_a2a_call_node(state: ClientAgentState) -> Dict[str, Any]:
            """Make a call to the A2A agent."""
            httpx_client = None
            try:
                # Get the original user query
                user_query = None
                for msg in state["messages"]:
                    if isinstance(msg, HumanMessage):
                        user_query = msg.content
                        break
                
                if not user_query:
                    return {
                        "messages": [AIMessage(content="Error: No user query found")],
                        "task_complete": True
                    }
                
                # Initialize A2A client and get both client and httpx_client
                client, httpx_client = await self.initialize_a2a_client()
                
                # Prepare the A2A request
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': user_query}],
                        'message_id': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                # Send message to A2A agent
                logger.info(f"Sending query to A2A agent: {user_query}")
                response = await client.send_message(request)
                
                # Extract the response content
                if response.root and response.root.result:
                    result_content = "A2A Agent Response received successfully"
                    
                    # Debug: log the response structure
                    logger.info(f"A2A Response structure: {type(response.root.result)}")
                    
                    # Handle Task object - this is the actual response structure
                    task_result = response.root.result
                    
                    # Dump the entire task structure to see what's inside
                    try:
                        task_dict = task_result.model_dump()
                        logger.info(f"Full Task structure: {task_dict}")
                    except Exception as e:
                        logger.warning(f"Could not dump task structure: {e}")
                        try:
                            # Try dict() method instead
                            task_dict = task_result.dict()
                            logger.info(f"Full Task structure (dict): {task_dict}")
                        except Exception as e2:
                            logger.warning(f"Could not use dict() either: {e2}")
                    
                    # Check artifacts more thoroughly
                    logger.info(f"Task has artifacts attribute: {hasattr(task_result, 'artifacts')}")
                    if hasattr(task_result, 'artifacts'):
                        artifacts = task_result.artifacts
                        logger.info(f"Artifacts value: {artifacts}")
                        logger.info(f"Artifacts type: {type(artifacts)}")
                        logger.info(f"Artifacts is None: {artifacts is None}")
                        if artifacts:
                            logger.info(f"Found {len(artifacts)} artifacts in Task")
                            for i, artifact in enumerate(artifacts):
                                logger.info(f"Artifact {i}: {type(artifact)}")
                                
                                # Check for parts within artifact
                                if hasattr(artifact, 'parts') and artifact.parts:
                                    logger.info(f"Artifact {i} has {len(artifact.parts)} parts")
                                    for j, part in enumerate(artifact.parts):
                                        logger.info(f"Part {j}: {type(part)}")
                                        
                                        # Try different ways to access the text
                                        if hasattr(part, 'root'):
                                            if hasattr(part.root, 'text'):
                                                result_content = part.root.text
                                                logger.info(f"Found text in part.root.text: {result_content[:100]}...")
                                                break
                                            elif hasattr(part.root, 'content'):
                                                result_content = part.root.content
                                                logger.info(f"Found text in part.root.content: {result_content[:100]}...")
                                                break
                                        
                                        # Try direct text access on part
                                        if hasattr(part, 'text'):
                                            result_content = part.text
                                            logger.info(f"Found text in part.text: {result_content[:100]}...")
                                            break
                                        
                                        # Try content field on part
                                        if hasattr(part, 'content'):
                                            result_content = part.content
                                            logger.info(f"Found text in part.content: {result_content[:100]}...")
                                            break
                                
                                # Try direct text access on artifact
                                if result_content == "A2A Agent Response received successfully":
                                    if hasattr(artifact, 'text'):
                                        result_content = artifact.text
                                        logger.info(f"Found text in artifact.text: {result_content[:100]}...")
                                        break
                                    elif hasattr(artifact, 'content'):
                                        result_content = artifact.content
                                        logger.info(f"Found text in artifact.content: {result_content[:100]}...")
                                        break
                    
                    # Try to extract from status.message (this is where responses are often stored)
                    if result_content == "A2A Agent Response received successfully" and hasattr(task_result, 'status'):
                        logger.info("Checking status.message for response content...")
                        status = task_result.status
                        if hasattr(status, 'message') and status.message:
                            message = status.message
                            logger.info(f"Found status message: {type(message)}")
                            
                            # Check message parts
                            if hasattr(message, 'parts') and message.parts:
                                logger.info(f"Message has {len(message.parts)} parts")
                                for i, part in enumerate(message.parts):
                                    logger.info(f"Message part {i}: {type(part)}")
                                    if hasattr(part, 'text') and part.text:
                                        result_content = part.text
                                        logger.info(f"Found response in status.message.parts[{i}].text: {result_content[:100]}...")
                                        break
                                    elif hasattr(part, 'content') and part.content:
                                        result_content = part.content
                                        logger.info(f"Found response in status.message.parts[{i}].content: {result_content[:100]}...")
                                        break
                                    # Try accessing the part directly - it might be a Pydantic object
                                    try:
                                        if hasattr(part, 'model_dump'):
                                            part_dict = part.model_dump()
                                            if 'text' in part_dict and part_dict['text']:
                                                result_content = part_dict['text']
                                                logger.info(f"Found response in part model_dump text: {result_content[:100]}...")
                                                break
                                    except:
                                        pass
                    
                    # If still no content found, try Task-level fields
                    if result_content == "A2A Agent Response received successfully":
                        logger.info("Trying Task-level fields...")
                        for attr in ['content', 'text', 'message', 'response', 'result']:
                            if hasattr(task_result, attr):
                                value = getattr(task_result, attr)
                                if value and isinstance(value, str):
                                    result_content = value
                                    logger.info(f"Found text in task_result.{attr}: {result_content[:100]}...")
                                    break
                    
                    # Last resort: convert the entire task to string and look for meaningful content
                    if result_content == "A2A Agent Response received successfully":
                        logger.info("Last resort: examining task structure...")
                        logger.info(f"Task attributes: {dir(task_result)}")
                        
                        # Print first few attributes that might contain content
                        for attr in dir(task_result):
                            if not attr.startswith('_'):
                                try:
                                    value = getattr(task_result, attr)
                                    if isinstance(value, str) and len(value) > 10:
                                        logger.info(f"Task.{attr}: {str(value)[:100]}...")
                                except:
                                    pass
                    
                    # Debug: show what we found
                    logger.info(f"Final extracted content: {result_content[:200]}...")
                    
                    return {
                        "messages": [AIMessage(content=f"A2A Response: {result_content}")],
                        "a2a_response": response,
                        "task_complete": True
                    }
                else:
                    return {
                        "messages": [AIMessage(content="Error: Invalid response from A2A agent")],
                        "task_complete": True
                    }
                    
            except Exception as e:
                logger.error(f"Error in A2A call: {e}")
                return {
                    "messages": [AIMessage(content=f"Error communicating with A2A agent: {str(e)}")],
                    "task_complete": True
                }
            finally:
                # Always close the httpx client to prevent resource leaks
                if httpx_client:
                    await httpx_client.aclose()
        
        async def direct_response_node(state: ClientAgentState) -> Dict[str, Any]:
            """Provide a direct response without using A2A agent."""
            # Get the original user query
            user_query = None
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    user_query = msg.content
                    break
            
            direct_prompt = f"""
            The user asked: "{user_query}"
            
            Provide a helpful direct response. Keep it concise and informative.
            """
            
            response = await self.client_llm.ainvoke([HumanMessage(content=direct_prompt)])
            
            return {
                "messages": [AIMessage(content=response.content)],
                "task_complete": True
            }
        
        def route_decision(state: ClientAgentState) -> str:
            """Route based on analysis decision."""
            last_message = state["messages"][-1]
            if "USE_A2A:" in last_message.content:
                return "use_a2a"
            elif "DIRECT:" in last_message.content:
                return "direct_response"
            else:
                return "direct_response"  # Default to direct response
        
        def should_end(state: ClientAgentState) -> str:
            """Check if task is complete."""
            if state.get("task_complete", False):
                return END
            return "continue"
        
        # Build the graph
        graph = StateGraph(ClientAgentState)
        
        # Add nodes
        graph.add_node("analyze_query", analyze_query_node)
        graph.add_node("make_a2a_call", make_a2a_call_node)
        graph.add_node("direct_response", direct_response_node)
        
        # Set entry point
        graph.set_entry_point("analyze_query")
        
        # Add conditional edges
        graph.add_conditional_edges(
            "analyze_query",
            route_decision,
            {
                "use_a2a": "make_a2a_call",
                "direct_response": "direct_response"
            }
        )
        
        # Add edges to end
        graph.add_conditional_edges(
            "make_a2a_call",
            should_end,
            {END: END, "continue": "analyze_query"}
        )
        
        graph.add_conditional_edges(
            "direct_response", 
            should_end,
            {END: END, "continue": "analyze_query"}
        )
        
        return graph.compile()
    
    async def process_query(self, query: str):
        """Process a user query through the client agent graph."""
        graph = self.build_client_graph()
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "task_complete": False
        }
        
        logger.info(f"Processing query: {query}")
        
        # Stream through the graph
        async for output in graph.astream(initial_state):
            for node_name, node_output in output.items():
                logger.info(f"Node '{node_name}' output: {node_output.get('messages', [])[-1].content if node_output.get('messages') else 'No message'}")
        
        # Get final state
        final_state = await graph.ainvoke(initial_state)
        
        return final_state


async def interactive_chat():
    """Interactive chat mode for asking questions to the client agent."""
    print("\n" + "="*60)
    print("🤖 INTERACTIVE A2A CLIENT AGENT")
    print("="*60)
    print("Ask me anything! I'll analyze your question and either:")
    print("• Answer directly for simple queries")
    print("• Use the A2A agent for complex research queries")
    print("\nType 'quit', 'exit', or 'bye' to stop.")
    print("Type 'test' to run predefined test queries.")
    print("-"*60)
    
    try:
        client_agent = SimpleA2AClientAgent()
        print("✅ Client agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize client agent: {e}")
        print("Please check your .env file and make sure the A2A server is running.")
        return
    
    while True:
        try:
            # Get user input
            user_input = input("\n💬 Your question: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("👋 Goodbye!")
                break
            
            # Check for test command
            if user_input.lower() == 'test':
                await run_test_queries(client_agent)
                continue
            
            # Skip empty inputs
            if not user_input:
                continue
            
            print(f"\n🔄 Processing your question...")
            
            # Process the query
            start_time = asyncio.get_event_loop().time()
            result = await client_agent.process_query(user_input)
            end_time = asyncio.get_event_loop().time()
            
            # Extract and display the final response
            if result and result.get('messages'):
                final_message = result['messages'][-1]
                response_content = final_message.content
                
                # Clean up the response format
                if response_content.startswith("A2A Response: "):
                    response_content = response_content[14:]  # Remove "A2A Response: " prefix
                    source = "🌐 A2A Agent"
                elif response_content.startswith("Analysis: "):
                    # This shouldn't be the final response, but handle it just in case
                    response_content = "Processing your request..."
                    source = "🤔 Analyzing"
                else:
                    source = "💭 Direct Response"
                
                print(f"\n{source}:")
                print("-" * 40)
                print(f"{response_content}")
                print("-" * 40)
                print(f"⏱️  Response time: {end_time - start_time:.2f}s")
            else:
                print("❌ No response received. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different question.")


async def run_test_queries(client_agent):
    """Run predefined test queries."""
    test_queries = [
        ("Simple Math", "What is 2+2?"),
        ("Current Events", "What are the latest developments in artificial intelligence?"),
        ("Academic Research", "Find recent papers on transformer architectures"),
        ("Greeting", "Hello, how are you?"),
        ("Complex Research", "I want to understand the recent breakthroughs in large language models and their applications")
    ]
    
    print(f"\n🧪 RUNNING {len(test_queries)} TEST QUERIES")
    print("="*60)
    
    for i, (category, query) in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] {category}: {query}")
        print("-" * 50)
        
        try:
            start_time = asyncio.get_event_loop().time()
            result = await client_agent.process_query(query)
            end_time = asyncio.get_event_loop().time()
            
            if result and result.get('messages'):
                final_response = result['messages'][-1].content
                if final_response.startswith("A2A Response: "):
                    final_response = final_response[14:]
                    route = "🌐 A2A"
                else:
                    route = "💭 Direct"
                
                print(f"Route: {route}")
                print(f"Response: {final_response[:200]}{'...' if len(final_response) > 200 else ''}")
                print(f"Time: {end_time - start_time:.2f}s")
            else:
                print("❌ No response received")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        if i < len(test_queries):  # Don't wait after the last query
            await asyncio.sleep(1)
    
    print(f"\n✅ Test completed!")


async def test_client_agent():
    """Legacy test function - now calls the interactive chat."""
    await interactive_chat()


def ask_question(question: str):
    """Simple synchronous wrapper to ask a single question."""
    async def _ask():
        try:
            client_agent = SimpleA2AClientAgent()
            result = await client_agent.process_query(question)
            
            if result and result.get('messages'):
                final_response = result['messages'][-1].content
                if final_response.startswith("A2A Response: "):
                    final_response = final_response[14:]
                return final_response
            else:
                return "No response received."
        except Exception as e:
            return f"Error: {e}"
    
    return asyncio.run(_ask())


if __name__ == "__main__":
    # Run the interactive chat interface
    asyncio.run(interactive_chat())