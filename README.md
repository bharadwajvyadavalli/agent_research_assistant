# Agent Research Assistant

A self-improving, multi-agent LLM system designed to assist with research paper analysis and complex problem-solving.

## Overview

This project implements an advanced AI agent system based on the principles outlined in "Architecting Self-Improving LLM Agent Systems for Complex Task Execution". The system uses a multi-agent architecture with specialized roles (Planner, Executor, Critic), sophisticated reasoning mechanisms (Chain of Thought, Tree of Thoughts), structured memory systems, and the Model Context Protocol (MCP) for tool integration.

Key features include:

- **Multi-Agent Architecture**: Specialized agents work together to tackle complex tasks
- **Advanced Reasoning**: Chain-of-Thought and Tree-of-Thoughts reasoning for complex problem-solving
- **Hybrid Memory System**: Episodic, semantic, and procedural memory for context retention 
- **Standardized Tool Integration**: MCP protocol for seamless integration with external tools
- **Self-Improvement**: Autonomous improvement through feedback and critic evaluation
- **PDF Analysis**: Built-in tools for analyzing academic papers and research documents

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agent-research-assistant.git
cd agent-research-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Interactive Mode

Run the assistant in interactive mode:

```bash
python main.py
```

In interactive mode, you can enter queries directly or use the following commands:
- `!pdf <path>`: Process a PDF document
- `!exit`: Exit the application

### Single Query Mode

Process a single query and exit:

```bash
python main.py --query "Explain the benefits of multi-agent LLM systems"
```

### PDF Processing

Process a PDF file with an optional query:

```bash
python main.py --pdf "path/to/document.pdf" --query "Summarize the key findings"
```

### API Mode

Run the assistant as an API server:

```bash
python main.py --api --host localhost --port 8000
```

The API exposes the following endpoints:
- `POST /query`: Process a query (JSON body: `{"query": "your query", "context": {}}`)
- `POST /pdf`: Process a PDF file (JSON body: `{"pdf_path": "path/to/file.pdf", "query": "optional query"}`)

## Architecture

The system consists of the following main components:

1. **Agents**:
   - **Planner Agent**: Creates structured plans for complex tasks
   - **Executor Agent**: Carries out specific tasks using available tools
   - **Critic Agent**: Evaluates outputs and provides feedback

2. **Reasoning**:
   - **Chain of Thought**: Linear, step-by-step reasoning
   - **Tree of Thoughts**: Exploring multiple reasoning paths simultaneously

3. **Memory**:
   - **Episodic Memory**: Stores specific events and experiences
   - **Semantic Memory**: Stores facts and concepts
   - **Vector Store**: Enables similarity-based memory retrieval

4. **MCP (Model Context Protocol)**:
   - **Client**: Connects to tool servers
   - **Server**: Exposes tools via standardized interfaces
   - **Tool Implementations**: E.g., PDF Reader

5. **Orchestration**:
   - **Orchestrator**: Coordinates agent interactions and workflow
   - **Feedback Loop**: Enables self-improvement

## Creating Custom Tools

You can create custom tools by implementing new MCP tool servers. For example:

```python
from mcp.protocol import MCPTool
from mcp.server import StandaloneToolServer

class MyCustomTool:
    @staticmethod
    async def execute_tool(param1, param