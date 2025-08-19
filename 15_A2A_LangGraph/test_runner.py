"""
Test runner for the A2A LangGraph implementations.

This script tests both the simple client agent and the persona agents
to demonstrate different ways of interacting with the A2A server.
"""
import asyncio
import logging
import os
from dotenv import load_dotenv

from client_agent import SimpleA2AClientAgent, test_client_agent
from persona_agent import PersonaAgent, test_persona_agents


# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_environment():
    """Check if required environment variables are set."""
    required_vars = ['OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file")
        return False
    
    return True


async def test_simple_client():
    """Test the simple client agent."""
    print("\n" + "="*60)
    print("TESTING SIMPLE CLIENT AGENT")
    print("="*60)
    
    try:
        await test_client_agent()
        print("✅ Simple client agent test completed successfully")
    except Exception as e:
        print(f"❌ Simple client agent test failed: {e}")


async def test_persona_agents_demo():
    """Test the persona agents."""
    print("\n" + "="*60)
    print("TESTING PERSONA AGENTS")
    print("="*60)
    
    try:
        await test_persona_agents()
        print("✅ Persona agents test completed successfully")
    except Exception as e:
        print(f"❌ Persona agents test failed: {e}")


async def interactive_demo():
    """Interactive demo allowing user to choose test type."""
    print("\n" + "="*60)
    print("INTERACTIVE A2A LANGGRAPH DEMO")
    print("="*60)
    
    print("\nChoose a test to run:")
    print("1. Simple Client Agent (basic A2A calls)")
    print("2. Persona Agents (advanced multi-persona testing)")
    print("3. Custom Query (test with your own query)")
    print("4. Run all tests")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                await test_simple_client()
            elif choice == "2":
                await test_persona_agents_demo()
            elif choice == "3":
                query = input("Enter your query: ").strip()
                if query:
                    client = SimpleA2AClientAgent()
                    result = await client.process_query(query)
                    print(f"\nResponse: {result['messages'][-1].content}")
            elif choice == "4":
                await test_simple_client()
                await test_persona_agents_demo()
            elif choice == "5":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def print_setup_instructions():
    """Print setup instructions for the user."""
    print("\n" + "="*60)
    print("A2A LANGGRAPH SETUP INSTRUCTIONS")
    print("="*60)
    
    print("""
To run these tests, you need to:

1. 🔑 Set up environment variables in .env file:
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here (optional, for web search)

2. 🚀 Start the A2A server in another terminal:
   uv run python -m app

3. ✅ Run this test script:
   python test_runner.py

4. 📊 Optional: View LangGraph Studio:
   uv run langgraph dev
   Then visit: https://smith.langchain.com/studio?baseUrl=http://localhost:2024

WHAT THIS DEMONSTRATES:
- ✨ Simple client agent that makes basic A2A calls
- 🎭 Persona agents with different research styles and satisfaction thresholds  
- 🔄 Multi-turn conversations through A2A protocol
- 📈 Helpfulness evaluation and iterative improvement
- 🛠️ Different LangGraph architectures for agent communication

The persona agents are particularly interesting as they demonstrate:
- Machine Learning Expert: Demands technical accuracy and sources
- Curious Student: Asks follow-up questions and wants to understand deeply
- Skeptical Researcher: Questions everything and has high standards
- Business Executive: Focuses on practical applications and business value
""")


async def main():
    """Main function to run the test suite."""
    print_setup_instructions()
    
    if not check_environment():
        print("\n❌ Environment setup incomplete. Please fix the above issues and try again.")
        return
    
    print("\n✅ Environment check passed!")
    
    # Check if A2A server is likely running
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:10000/.well-known/agent_card")
            if response.status_code == 200:
                print("✅ A2A server appears to be running!")
            else:
                print("⚠️ A2A server might not be running. Start it with: uv run python -m app")
    except:
        print("⚠️ A2A server might not be running. Start it with: uv run python -m app")
    
    await interactive_demo()


if __name__ == "__main__":
    asyncio.run(main())