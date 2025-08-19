"""
Advanced LangGraph Agent with Different Personas for A2A Testing.

This creates different AI personas that test the A2A agent with various goals,
demonstrating how different agents can interact through the A2A protocol.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Annotated, TypedDict, Optional
from uuid import uuid4

import httpx
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonaAgentState(TypedDict):
    """State schema for persona agents."""
    messages: Annotated[List, add_messages]
    persona: str
    goal: str
    iteration_count: int
    satisfaction_level: int  # 1-5 scale
    gathered_information: List[str]
    follow_up_questions: List[str]
    task_complete: bool


class PersonaAgent:
    """An AI agent with different personas that test the A2A agent."""
    
    PERSONAS = {
        "machine_learning_expert": {
            "name": "Dr. Sarah Chen - ML Research Expert",
            "description": "You are an expert in Machine Learning with a PhD from Stanford. You are meticulous about sources, demand technical accuracy, and are not satisfied with surface-level answers. You always want to verify information with academic sources.",
            "satisfaction_threshold": 4,
            "max_iterations": 3
        },
        "curious_student": {
            "name": "Alex Rivera - Eager Student", 
            "description": "You are an enthusiastic computer science student who loves learning about new technologies. You ask follow-up questions and want to understand both the basics and advanced concepts. You're excited about AI developments.",
            "satisfaction_threshold": 3,
            "max_iterations": 2
        },
        "business_executive": {
            "name": "Michael Thompson - Tech Executive",
            "description": "You are a business executive evaluating AI technologies for your company. You focus on practical applications, business value, and market trends. You need clear, actionable insights rather than technical details.",
            "satisfaction_threshold": 3,
            "max_iterations": 2
        },
        "skeptical_researcher": {
            "name": "Dr. Elena Volkov - Critical Researcher",
            "description": "You are a highly skeptical researcher who questions everything. You demand multiple sources, look for limitations and biases, and are never easily satisfied. You want to understand the methodology behind claims.",
            "satisfaction_threshold": 5,
            "max_iterations": 4
        }
    }
    
    def __init__(self, persona_key: str, a2a_server_url: str = "http://localhost:10000"):
        if persona_key not in self.PERSONAS:
            raise ValueError(f"Unknown persona: {persona_key}")
            
        self.persona_key = persona_key
        self.persona_config = self.PERSONAS[persona_key]
        self.a2a_server_url = a2a_server_url
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7
        )
    
    async def initialize_a2a_client(self) -> A2AClient:
        """Initialize the A2A client."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=self.a2a_server_url
            )
            
            agent_card = await resolver.get_agent_card()
            return A2AClient(httpx_client=httpx_client, agent_card=agent_card)
    
    def build_persona_graph(self):
        """Build the LangGraph for this persona agent."""
        
        async def formulate_query_node(state: PersonaAgentState) -> Dict[str, Any]:
            """Formulate an initial query based on persona and goal."""
            persona_desc = self.persona_config["description"]
            goal = state["goal"]
            
            if state["iteration_count"] == 0:
                # Initial query
                prompt = f"""
                {persona_desc}
                
                Your research goal is: {goal}
                
                Formulate an initial query that aligns with your persona and research interests. 
                Make it specific enough to get valuable information, but open enough to allow for follow-up questions.
                
                Return only the query, no explanation.
                """
            else:
                # Follow-up query based on previous information
                gathered_info = "\n".join(state["gathered_information"])
                
                prompt = f"""
                {persona_desc}
                
                Your research goal is: {goal}
                
                Previous information gathered:
                {gathered_info}
                
                Based on your persona and the information already gathered, formulate a follow-up question that:
                1. Digs deeper into the topic
                2. Addresses any gaps or concerns you might have
                3. Reflects your specific expertise and standards
                
                Return only the query, no explanation.
                """
            
            response = await self.llm.ainvoke([
                SystemMessage(content=persona_desc),
                HumanMessage(content=prompt)
            ])
            
            return {
                "messages": [HumanMessage(content=response.content)]
            }
        
        async def query_a2a_agent_node(state: PersonaAgentState) -> Dict[str, Any]:
            """Send query to the A2A agent and get response."""
            try:
                # Get the query from the last message
                query = state["messages"][-1].content
                
                # Initialize A2A client
                client = await self.initialize_a2a_client()
                
                # Prepare A2A request
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': query}],
                        'message_id': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                logger.info(f"[{self.persona_config['name']}] Querying A2A agent: {query}")
                response = await client.send_message(request)
                
                # Extract response content
                response_content = "No response received"
                if response.root and response.root.result:
                    if hasattr(response.root.result, 'artifacts') and response.root.result.artifacts:
                        for artifact in response.root.result.artifacts:
                            if hasattr(artifact, 'parts') and artifact.parts:
                                for part in artifact.parts:
                                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                        response_content = part.root.text
                                        break
                
                # Update gathered information
                new_gathered_info = state["gathered_information"] + [f"Q: {query}\nA: {response_content}"]
                
                return {
                    "messages": [AIMessage(content=response_content)],
                    "gathered_information": new_gathered_info
                }
                
            except Exception as e:
                logger.error(f"Error querying A2A agent: {e}")
                return {
                    "messages": [AIMessage(content=f"Error: {str(e)}")],
                    "gathered_information": state["gathered_information"] + [f"Error occurred: {str(e)}"]
                }
        
        async def evaluate_satisfaction_node(state: PersonaAgentState) -> Dict[str, Any]:
            """Evaluate satisfaction with the received information."""
            persona_desc = self.persona_config["description"]
            goal = state["goal"]
            gathered_info = "\n".join(state["gathered_information"])
            
            evaluation_prompt = f"""
            {persona_desc}
            
            Your research goal is: {goal}
            
            Information gathered so far:
            {gathered_info}
            
            Evaluate your satisfaction with the information received on a scale of 1-5:
            1 = Very unsatisfied, major gaps or quality issues
            2 = Somewhat unsatisfied, needs more detail or sources
            3 = Neutral, adequate but could be better
            4 = Satisfied, good information that meets most needs
            5 = Very satisfied, excellent comprehensive information
            
            Also, if you're not fully satisfied, suggest 1-2 specific follow-up questions you would ask.
            
            Format your response as:
            SATISFACTION: [number]
            REASONING: [brief explanation]
            FOLLOW_UP: [questions, if any]
            """
            
            response = await self.llm.ainvoke([
                SystemMessage(content=persona_desc),
                HumanMessage(content=evaluation_prompt)
            ])
            
            # Parse the response
            response_text = response.content
            satisfaction_level = 3  # default
            follow_ups = []
            
            try:
                lines = response_text.split('\n')
                for line in lines:
                    if line.startswith('SATISFACTION:'):
                        satisfaction_level = int(line.split(':')[1].strip())
                    elif line.startswith('FOLLOW_UP:') and ':' in line:
                        follow_up_text = line.split(':', 1)[1].strip()
                        if follow_up_text and follow_up_text != 'None':
                            follow_ups = [follow_up_text]
            except:
                pass
            
            return {
                "messages": [AIMessage(content=response_text)],
                "satisfaction_level": satisfaction_level,
                "follow_up_questions": follow_ups
            }
        
        def should_continue(state: PersonaAgentState) -> str:
            """Decide whether to continue asking questions or end."""
            satisfaction_threshold = self.persona_config["satisfaction_threshold"]
            max_iterations = self.persona_config["max_iterations"]
            
            current_satisfaction = state.get("satisfaction_level", 0)
            current_iteration = state.get("iteration_count", 0)
            
            # Check if satisfied or reached max iterations
            if current_satisfaction >= satisfaction_threshold or current_iteration >= max_iterations:
                return "end"
            
            # Check if there are follow-up questions
            if state.get("follow_up_questions"):
                return "continue"
            
            return "end"
        
        async def prepare_next_iteration_node(state: PersonaAgentState) -> Dict[str, Any]:
            """Prepare for the next iteration."""
            return {
                "iteration_count": state["iteration_count"] + 1,
                "messages": []  # Clear messages for next query
            }
        
        # Build the graph
        graph = StateGraph(PersonaAgentState)
        
        # Add nodes
        graph.add_node("formulate_query", formulate_query_node)
        graph.add_node("query_a2a_agent", query_a2a_agent_node)
        graph.add_node("evaluate_satisfaction", evaluate_satisfaction_node)
        graph.add_node("prepare_next_iteration", prepare_next_iteration_node)
        
        # Set entry point
        graph.set_entry_point("formulate_query")
        
        # Add edges
        graph.add_edge("formulate_query", "query_a2a_agent")
        graph.add_edge("query_a2a_agent", "evaluate_satisfaction")
        
        # Add conditional edges
        graph.add_conditional_edges(
            "evaluate_satisfaction",
            should_continue,
            {
                "continue": "prepare_next_iteration",
                "end": END
            }
        )
        
        graph.add_edge("prepare_next_iteration", "formulate_query")
        
        return graph.compile()
    
    async def research_topic(self, goal: str):
        """Research a topic using the persona's approach."""
        graph = self.build_persona_graph()
        
        initial_state = PersonaAgentState(
            messages=[],
            persona=self.persona_key,
            goal=goal,
            iteration_count=0,
            satisfaction_level=0,
            gathered_information=[],
            follow_up_questions=[],
            task_complete=False
        )
        
        logger.info(f"[{self.persona_config['name']}] Starting research on: {goal}")
        
        result = await graph.ainvoke(initial_state)
        
        return {
            "persona": self.persona_config["name"],
            "goal": goal,
            "final_satisfaction": result.get("satisfaction_level", 0),
            "iterations": result.get("iteration_count", 0),
            "gathered_information": result.get("gathered_information", []),
            "threshold": self.persona_config["satisfaction_threshold"]
        }


async def test_persona_agents():
    """Test different persona agents with various research goals."""
    
    research_topics = [
        "I want to learn about what makes large language models so incredible, particularly focusing on recent breakthroughs and technical innovations",
        "Recent developments in transformer architecture and their impact on natural language processing",
        "The business applications and market potential of artificial intelligence in 2024"
    ]
    
    personas_to_test = ["machine_learning_expert", "curious_student", "skeptical_researcher"]
    
    for i, topic in enumerate(research_topics):
        persona_key = personas_to_test[i % len(personas_to_test)]
        
        print(f"\n{'='*80}")
        print(f"TESTING PERSONA: {PersonaAgent.PERSONAS[persona_key]['name']}")
        print(f"RESEARCH GOAL: {topic}")
        print('='*80)
        
        try:
            persona_agent = PersonaAgent(persona_key)
            result = await persona_agent.research_topic(topic)
            
            print(f"\nRESULT SUMMARY:")
            print(f"- Final Satisfaction: {result['final_satisfaction']}/{result['threshold']}")
            print(f"- Iterations: {result['iterations']}")
            print(f"- Information Gathered: {len(result['gathered_information'])} exchanges")
            
            print(f"\nDETAILED INFORMATION GATHERED:")
            for info in result['gathered_information']:
                print(f"\n{'-'*40}")
                print(info)
            
        except Exception as e:
            print(f"Error testing persona {persona_key}: {e}")
        
        # Wait between tests
        await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(test_persona_agents())