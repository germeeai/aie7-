# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
uv sync                          # Install all dependencies
```

### Guardrails Configuration

**Note**: Dependencies were missing and have been resolved. Run these commands to verify setup:

```bash
uv run guardrails configure      # Configure Guardrails API key
uv add huggingface_hub          # Add HuggingFace Hub for jailbreak guards
uv run huggingface-cli login    # Login to HuggingFace
uv run guardrails hub list      # Verify installed guards
```

**Required Dependencies** (automatically added to pyproject.toml):
- `alt-profanity-check>=1.7.1` (replaces broken profanity-check)
- `presidio-anonymizer>=2.2.359` (for PII detection)
- `presidio-analyzer>=2.2.359` (for PII analysis)
- `gliner>=0.2.21` (for named entity recognition)
- `joblib>=1.5.1` (for sklearn compatibility)

### Install Guardrails Guards
```bash
uv run guardrails hub install hub://tryolabs/restricttotopic
uv run guardrails hub install hub://guardrails/detect_jailbreak
uv run guardrails hub install hub://guardrails/competitor_check
uv run guardrails hub install hub://arize-ai/llm_rag_evaluator
uv run guardrails hub install hub://guardrails/profanity_free
uv run guardrails hub install hub://guardrails/guardrails_pii
```

### Jupyter Notebook
```bash
jupyter notebook                 # Start Jupyter for the assignment notebook
```

## Architecture Overview

### Core Library Structure (`langgraph_agent_lib/`)

**ProductionRAGChain** (`rag.py`):
- Production-ready RAG implementation with caching
- Uses PyMuPDF for PDF loading, Qdrant for vector storage
- Implements MMR retrieval and in-memory vector storage
- Key parameters: chunk_size (1000), chunk_overlap (100)

**CacheBackedEmbeddings** (`caching.py`):
- File-based caching for OpenAI embeddings using LocalFileStore
- Supports both in-memory and SQLite LLM caching
- Uses MD5 hashing for safe cache namespacing

**LangGraph Agent** (`agents.py`):
- Implements state-based agent using LangGraph StateGraph
- Integrates Tavily search, Arxiv, and RAG tools
- Uses conditional routing between agent and tool execution nodes

**Model Utilities** (`models.py`):
- Centralized OpenAI model configuration
- Default model: "gpt-4.1-mini" with temperature 0.1

### Data Processing Pipeline
1. PDF documents loaded from `data/` directory
2. Cached embeddings stored in `cache/embeddings/`
3. Vector storage uses in-memory Qdrant with COSINE distance
4. Agent integrates RAG retrieval with web search and arxiv tools

### Environment Requirements
- Python 3.11.13 (exact version required)
- OpenAI API key for LLM and embeddings
- Tavily API key for web search (optional)
- Guardrails AI API key for safety checks
- HuggingFace token for jailbreak detection

### Known Issues & Workarounds

**PII Guard Tokenizer Issue**: The local GLiNER model has tokenizer conversion issues. Use cloud-based detection:
```python
# ❌ This will fail with tokenizer conversion error:
pii_guard = Guard().use(GuardrailsPII(entities=["CREDIT_CARD"], use_local=True))

# ✅ Use this instead:
pii_guard = Guard().use(GuardrailsPII(entities=["CREDIT_CARD"], use_local=False))
```

### Assignment Workflow
The main assignment is in `Prototyping_LangChain_Application_with_Production_Minded_Changes_Assignment.ipynb` which implements:
1. Production RAG setup with caching
2. LangGraph agent integration
3. Guardrails integration for content safety
4. Performance optimization through caching layers

### Activity #3: Production-Safe Agent Implementation

**Core Files**:
- `langgraph_agent_lib/guardrails_nodes.py` - Guardrails validation nodes
- `langgraph_agent_lib/safe_agent.py` - Production-safe LangGraph agent
- `production_safe_agent_demo.py` - Complete demonstration script
- `test_safe_agent.py` - Adversarial test suite

**Key Components Implemented**:
1. **Input Validation Node**: Jailbreak detection, topic validation, PII screening
2. **Output Validation Node**: Content moderation, competitor filtering, PII protection
3. **Refinement Loop**: Automatic retry mechanism for failed validations (max 3 attempts)
4. **Conditional Routing**: LangGraph-based flow control for validation decisions
5. **Comprehensive Logging**: Security event tracking and monitoring

**Usage Example**:
```python
from langgraph_agent_lib import SafeAgentConfig, create_safe_agent

config = SafeAgentConfig(
    allowed_topics=["student loans", "financial aid"],
    competitors=["OpenAI", "ChatGPT"],
    pii_entities=["CREDIT_CARD", "SSN"],
    max_refinements=3
)

agent = create_safe_agent(config)
result = agent.invoke("What are federal student loans?")
```