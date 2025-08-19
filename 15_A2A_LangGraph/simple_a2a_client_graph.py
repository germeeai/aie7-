"""
Simple A2A Client Graph - Assignment Implementation

This creates a LangGraph that acts as a simple agent making API calls to the 
A2A Agent Node through the A2A protocol, following the assignment requirements.

Based on the structure of app/simple_a2a_graph.py but enhanced for better
functionality and user experience.
"""

import os
import asyncio
from typing import TypedDict, Annotated, List
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendStreamingMessageRequest

# Load environment variables
load_dotenv()


class SimpleA2AClientState(TypedDict):
    """State schema for the simple A2A client graph."""
    messages: Annotated[List, add_messages]
    a2a_server_url: str
    query_analysis: str
    use_a2a: bool


def create_simple_a2a_client_graph():
    """
    Create a simple LangGraph that uses the A2A application.
    
    This implements the assignment requirement: "Build a LangGraph Graph to 'use' 
    your application by creating a Simple Agent that can make API calls to the 
    🤖Agent Node above through the A2A protocol."
    """
    
    # Initialize the analysis LLM
    analysis_llm = ChatOpenAI(
        model=os.getenv('TOOL_LLM_NAME', 'gpt-4o-mini'),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        temperature=0.3
    )
    
    async def analyze_query_node(state: SimpleA2AClientState):
        """Analyze whether the query should use the A2A agent or be handled directly."""
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if not user_message:
            return {
                "messages": [AIMessage(content="No user message found")],
                "use_a2a": False
            }
        
        analysis_prompt = f"""
        Analyze this user query: "{user_message}"
        
        The A2A agent has these capabilities:
        - Web search for current information
        - Academic paper search on arXiv
        - Document retrieval from a knowledge base (RAG)
        - Complex multi-step reasoning
        
        Should this query use the A2A agent (respond with "YES" or "NO")?
        
        Use A2A if the query:
        - Needs current web information
        - Requires academic research
        - Involves document retrieval
        - Is complex and benefits from multiple tools
        
        Use direct response if the query:
        - Is simple math or basic facts
        - Is a greeting or casual conversation
        - Can be answered without external tools
        
        Respond with just "YES" or "NO" and a brief reason.
        """
        
        try:
            response = await analysis_llm.ainvoke([HumanMessage(content=analysis_prompt)])
            analysis = response.content
            
            use_a2a = "YES" in analysis.upper()
            
            return {
                "messages": [AIMessage(content=f"Analysis: {analysis}")],
                "query_analysis": analysis,
                "use_a2a": use_a2a
            }
        except Exception as e:
            # Default to using A2A on analysis error
            return {
                "messages": [AIMessage(content=f"Analysis error: {e}")],
                "query_analysis": "Error occurred, defaulting to A2A",
                "use_a2a": True
            }
    
    async def query_a2a_server_node(state: SimpleA2AClientState):
        """Query the A2A server using streaming API for better response extraction."""
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if not user_message:
            return {"messages": [AIMessage(content="No user message found")]}
        
        base_url = state.get("a2a_server_url", "http://localhost:10000")
        
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                # Get agent card
                resolver = A2ACardResolver(httpx_client=client, base_url=base_url)
                agent_card = await resolver.get_agent_card()
                print(f"✅ Connected to A2A agent: {agent_card.name}")
                
                # Create A2A client
                a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)
                
                # Prepare streaming request (this works better than regular requests)
                request = SendStreamingMessageRequest(
                    id=str(uuid4()),
                    params=MessageSendParams(
                        message={
                            'role': 'user',
                            'parts': [{'kind': 'text', 'text': user_message}],
                            'message_id': uuid4().hex,
                        }
                    )
                )
                
                print(f"🔄 Sending query to A2A agent: {user_message}")
                
                # Send streaming request and collect response
                stream_response = a2a_client.send_message_streaming(request)
                
                # Collect response from streaming chunks
                full_content = ""
                working_messages = []
                task_id = ""
                context_id = ""
                
                async for chunk in stream_response:
                    # Convert chunk to dict for easier access
                    chunk_dict = chunk.model_dump(mode='json', exclude_none=True)
                    
                    if 'result' in chunk_dict:
                        result = chunk_dict['result']
                        
                        # Get task/context IDs from first chunk
                        if 'id' in result and not task_id:
                            task_id = result['id']
                        if 'contextId' in result and not context_id:
                            context_id = result['contextId']
                        
                        # Handle different chunk types
                        if result.get('kind') == 'task-update':
                            # Status updates (working, searching, etc.)
                            if 'status' in result and 'message' in result['status']:
                                status_msg = result['status']['message']
                                if 'parts' in status_msg:
                                    for part in status_msg['parts']:
                                        if part.get('kind') == 'text' and 'text' in part:
                                            working_content = part['text']
                                            if working_content and "Searching" in working_content:
                                                print(f"🔍 {working_content}")
                                                working_messages.append(working_content)
                        
                        elif result.get('kind') == 'artifact-update':
                            # Final response content
                            artifact = result.get('artifact', {})
                            parts = artifact.get('parts', [])
                            for part in parts:
                                if part.get('kind') == 'text' and 'text' in part:
                                    full_content = part['text']
                                    print(f"✅ Received final response ({len(full_content)} chars)")
                                    break
                
                # Use the full content if available, otherwise show working messages
                if full_content:
                    response_content = full_content
                elif working_messages:
                    response_content = f"Working on your request...\n\n{working_messages[-1]}"
                else:
                    response_content = "Response received from A2A server but no content extracted."
                
                return {"messages": [AIMessage(content=response_content)]}
                
        except Exception as e:
            error_msg = f"Error communicating with A2A server: {str(e)}"
            print(f"❌ {error_msg}")
            return {"messages": [AIMessage(content=error_msg)]}
    
    async def direct_response_node(state: SimpleA2AClientState):
        """Provide a direct response without using the A2A agent."""
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if not user_message:
            return {"messages": [AIMessage(content="No user message found")]}
        
        try:
            direct_prompt = f"""
            Provide a helpful, direct response to this query: "{user_message}"
            
            Keep your response concise and informative. This is a simple query 
            that doesn't require web search or complex research.
            """
            
            response = await analysis_llm.ainvoke([HumanMessage(content=direct_prompt)])
            return {"messages": [AIMessage(content=response.content)]}
            
        except Exception as e:
            return {"messages": [AIMessage(content=f"Error generating direct response: {e}")]}
    
    def routing_decision(state: SimpleA2AClientState) -> str:
        """Route based on the query analysis."""
        if state.get("use_a2a", True):  # Default to A2A if unsure
            return "use_a2a"
        else:
            return "direct_response"
    
    # Build the graph
    workflow = StateGraph(SimpleA2AClientState)
    
    # Add nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("query_a2a_server", query_a2a_server_node)
    workflow.add_node("direct_response", direct_response_node)
    
    # Set entry point
    workflow.set_entry_point("analyze_query")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "analyze_query",
        routing_decision,
        {
            "use_a2a": "query_a2a_server",
            "direct_response": "direct_response"
        }
    )
    
    # Add edges to end
    workflow.add_edge("query_a2a_server", END)
    workflow.add_edge("direct_response", END)
    
    return workflow.compile()


async def interactive_demo():
    """Interactive demo of the simple A2A client graph."""
    print("\n" + "="*70)
    print("🤖 SIMPLE A2A CLIENT GRAPH - ASSIGNMENT IMPLEMENTATION")
    print("="*70)
    print("This LangGraph demonstrates using the A2A application through the A2A protocol.")
    print("\nFeatures:")
    print("• Intelligent query analysis (A2A vs direct response)")
    print("• Streaming A2A communication")
    print("• Real-time status updates")
    print("• Proper response extraction")
    print("\nType 'quit' to exit, 'test' to run test queries.")
    print("-"*70)
    
    # Create the graph
    try:
        graph = create_simple_a2a_client_graph()
        print("✅ Simple A2A Client Graph initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize graph: {e}")
        return
    
    while True:
        try:
            user_input = input("\n💬 Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'test':
                await run_test_queries(graph)
                continue
            
            if not user_input:
                continue
            
            print(f"\n🔄 Processing: {user_input}")
            print("-" * 50)
            
            # Run the graph
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "a2a_server_url": os.getenv('A2A_SERVER_URL', 'http://localhost:10000'),
                "use_a2a": True  # Will be determined by analysis
            }
            
            start_time = asyncio.get_event_loop().time()
            result = await graph.ainvoke(initial_state)
            end_time = asyncio.get_event_loop().time()
            
            # Show the final response
            final_message = result["messages"][-1]
            analysis_message = None
            
            # Find the analysis message
            for msg in result["messages"]:
                if msg.content.startswith("Analysis:"):
                    analysis_message = msg.content
                    break
            
            if analysis_message:
                print(f"🧠 {analysis_message}")
                print()
            
            print(f"📤 Final Response:")
            print(f"{final_message.content}")
            print(f"\n⏱️  Total time: {end_time - start_time:.2f}s")
            print("=" * 70)
            
        except KeyboardInterrupt:
            print("\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


async def run_test_queries(graph):
    """Run predefined test queries to demonstrate the graph."""
    test_queries = [
        ("Simple Math", "What is 15 * 7?"),
        ("Current Events", "What are the latest developments in AI?"),
        ("Academic Research", "Find recent papers on RAG evaluation"),
        ("RAG Question", "What's the first thing to do when evaluating RAG systems?"),
        ("Greeting", "Hello, how are you?")
    ]
    
    print(f"\n🧪 RUNNING {len(test_queries)} TEST QUERIES")
    print("="*70)
    
    for i, (category, query) in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] {category}: {query}")
        print("-" * 50)
        
        try:
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "a2a_server_url": "http://localhost:10000",
                "use_a2a": True
            }
            
            start_time = asyncio.get_event_loop().time()
            result = await graph.ainvoke(initial_state)
            end_time = asyncio.get_event_loop().time()
            
            # Show analysis and response
            for msg in result["messages"]:
                if msg.content.startswith("Analysis:"):
                    use_a2a = "A2A" if "YES" in msg.content else "Direct"
                    print(f"🧠 Route: {use_a2a}")
                    break
            
            final_response = result["messages"][-1].content
            print(f"📤 Response: {final_response[:150]}{'...' if len(final_response) > 150 else ''}")
            print(f"⏱️  Time: {end_time - start_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        if i < len(test_queries):
            await asyncio.sleep(1)  # Brief pause between tests
    
    print("\n✅ Test queries completed!")


async def main():
    """Main function to run the demo."""
    # Check prerequisites
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file.")
        return
    
    print("""
📋 Prerequisites Check:
1. ✅ Make sure the A2A server is running: uv run python -m app
2. ✅ Server should be available at http://localhost:10000
3. ✅ Environment variables loaded from .env file

🎯 This implements the assignment requirement:
"Build a LangGraph Graph to 'use' your application by creating a Simple Agent 
that can make API calls to the 🤖Agent Node above through the A2A protocol."
""")
    
    await interactive_demo()


if __name__ == "__main__":
    asyncio.run(main())