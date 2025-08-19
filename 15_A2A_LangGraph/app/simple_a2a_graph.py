"""
Simple A2A LangGraph Example
This demonstrates a minimal LangGraph that can communicate with the A2A server.
"""

import os
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
import httpx
import asyncio
from uuid import uuid4
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendStreamingMessageRequest


class SimpleA2AState(TypedDict):
    """State for our simple A2A graph"""
    messages: Annotated[List, add_messages]
    a2a_server_url: str
    
    
def create_simple_a2a_graph():
    """Create a simple graph that queries the A2A server"""
    
    async def query_a2a_server(state: SimpleA2AState):
        """Node that queries the A2A server"""
        # Get the last user message
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if not user_message:
            return {"messages": [AIMessage(content="No user message found")]}
        
        # Query the A2A server
        base_url = state.get("a2a_server_url", "http://localhost:10000")
        
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                # Get agent card
                resolver = A2ACardResolver(httpx_client=client, base_url=base_url)
                agent_card = await resolver.get_agent_card()
                
                # Create A2A client (note: this is deprecated but still works)
                a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)
                
                # Prepare streaming request
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
                
                # Send streaming request and collect response
                stream_response = a2a_client.send_message_streaming(request)
                
                # Collect response from chunks
                full_content = ""
                task_id = ""
                context_id = ""
                
                chunk_count = 0
                async for chunk in stream_response:
                    chunk_count += 1
                    
                    # Convert chunk to dict for easier access
                    chunk_dict = chunk.model_dump(mode='json', exclude_none=True)
                    
                    if 'result' in chunk_dict:
                        result = chunk_dict['result']
                        
                        # Get task/context IDs from first chunk
                        if 'id' in result and not task_id:
                            task_id = result['id']
                        if 'contextId' in result and not context_id:
                            context_id = result['contextId']
                        
                        # Look for artifact-update chunks which contain the actual response
                        if result.get('kind') == 'artifact-update':
                            artifact = result.get('artifact', {})
                            parts = artifact.get('parts', [])
                            for part in parts:
                                if part.get('kind') == 'text' and 'text' in part:
                                    full_content = part['text']
                                    break
                
                if not full_content:
                    full_content = "No response received from A2A server"
                
                return {"messages": [AIMessage(content=full_content)]}
                
        except Exception as e:
            return {"messages": [AIMessage(content=f"Error querying A2A server: {str(e)}")]}
    
    
    # Build the graph
    workflow = StateGraph(SimpleA2AState)
    
    # Add the single node
    workflow.add_node("query_a2a", query_a2a_server)
    
    # Set entry and exit
    workflow.set_entry_point("query_a2a")
    workflow.add_edge("query_a2a", END)
    
    # Compile
    return workflow.compile()


async def main():
    """Demo the simple A2A graph"""
    print("\n🎯 Simple A2A LangGraph Demo\n")
    
    # Create the graph
    graph = create_simple_a2a_graph()
    
    # Test queries
    test_queries = [
        "What is 2+2?",  # Simple query that should work
        "What is machine learning?",
        "Find recent papers on neural networks",
        "Search for information about transformers"
    ]
    
    for query in test_queries:
        print(f"\n📤 Sending query: {query}")
        print("-" * 50)
        
        # Run the graph
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "a2a_server_url": "http://localhost:10000"
        }
        
        try:
            result = await graph.ainvoke(initial_state)
            
            # Print the response
            last_message = result["messages"][-1]
            print(f"📥 {last_message.content}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("=" * 70)


if __name__ == "__main__":
    print("""
⚠️  Prerequisites:
1. Make sure you have the A2A server running: uv run python -m app
2. This will connect to http://localhost:10000

Press Ctrl+C to stop.
""")
    
    asyncio.run(main())