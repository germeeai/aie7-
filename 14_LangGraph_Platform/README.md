<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 14: Build & Serve Agentic Graphs with LangGraph</h1>

| 🤓 Pre-work | 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 14: Pre-Work](https://www.notion.so/Session-14-Deploying-Agents-to-Production-21dcd547af3d80aba092fcb6c649c150?source=copy_link#247cd547af3d80709683ff380f4cba62)| [Session 14: Deploying Agents to Production](https://www.notion.so/Session-14-Deploying-Agents-to-Production-21dcd547af3d80aba092fcb6c649c150) | [Recording!](https://us02web.zoom.us/rec/share/1YepNUK3kqQnYLY8InMfHv84JeiOMyjMRWOZQ9jfjY86dDPvHMhyoz5Zo04w_tn-.91KwoSPyP6K6u0DC)  (@@5J6DVQ)| [Session 14 Slides](https://www.canva.com/design/DAGvVPg7-mw/IRwoSgDXPEqU-PKeIw8zLg/edit?utm_content=DAGvVPg7-mw&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 14 Assignment: Production Agents](https://forms.gle/nZ7ugE4W9VsC1zXE8) | [AIE7 Feedback 8/7](https://forms.gle/juo8SF5y5XiojFyC9)

# Build 🏗️

Run the repository and complete the following:

- 🤝 Breakout Room Part #1 — Building and serving your LangGraph Agent Graph
  - Task 1: Getting Dependencies & Environment
    - Configure `.env` (OpenAI, Tavily, optional LangSmith)
  - Task 2: Serve the Graph Locally
    - `uv run langgraph dev` (API on http://localhost:2024)
  - Task 3: Call the API
    - `uv run test_served_graph.py` (sync SDK example)
  - Task 4: Explore assistants (from `langgraph.json`)
    - `agent` → `simple_agent` (tool-using agent)
    - `agent_helpful` → `agent_with_helpfulness` (separate helpfulness node)

- 🤝 Breakout Room Part #2 — Using LangGraph Studio to visualize the graph
  - Task 1: Open Studio while the server is running
    - https://smith.langchain.com/studio?baseUrl=http://localhost:2024
  - Task 2: Visualize & Stream
    - Start a run and observe node-by-node updates
  - Task 3: Compare Flows
    - Contrast `agent` vs `agent_helpful` (tool calls vs helpfulness decision)

<details>
<summary>🚧 Advanced Build 🚧 (OPTIONAL - <i>open this section for the requirements</i>)</summary>

- Create and deploy a locally hosted MCP server with FastMCP.
- Extend your tools in `tools.py` to allow your LangGraph to consume the MCP Server.
</details>

# Ship 🚢

- Running local server (`langgraph dev`)
- Short demo showing both assistants responding

# Share 🚀
- Walk through your graph in Studio
- Share 3 lessons learned and 3 lessons not learned


#### ❓ Question:

What is the purpose of the `chunk_overlap` parameter when using `RecursiveCharacterTextSplitter` to prepare documents for RAG, and what trade-offs arise as you increase or decrease its value?

The chunk_overlap parameter in RecursiveCharacterTextSplitter controls how many characters (or tokens, depending on length_function) from the end of one chunk are repeated at the start of the next chunk when splitting text for RAG.

Higher overlap improves context recall but costs more in storage and processing, while lower overlap is more efficient but may hurt retrieval quality.


#### ❓ Question:

Your retriever is configured with `search_kwargs={"k": 5}`. How would adjusting `k` likely affect RAGAS metrics such as Context Precision and Context Recall in practice, and why?

Context Precision - Likely increases when k is fewer because you’re retrieving more relevant documents, making the context more focused
Context Recall - Often decreases when k is fewer because you risk missing some relevant documents if they don’t make the top spots


#### ❓ Question:

Compare the `agent` and `agent_helpful` assistants defined in `langgraph.json`. Where does the helpfulness evaluator fit in the graph, and under what condition should execution route back to the agent vs. terminate?

Comparison: agent uses the simple_agent graph (model ↔ tools loop only). agent_helpful uses agent_with_helpfulness, which adds a helpfulness evaluator step after the model/tool phase

After the model responds (and there are no tool calls), execution goes to the helpfulness node to judge the reply

If the evaluator outputs HELPFULNESS:Y (or the loop-limit is hit → HELPFULNESS:END), the run terminates; if it outputs HELPFULNESS:N, execution routes back to the agent node to try again