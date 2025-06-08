# ContextKernel 🧠

**Give your AI a real brain. ContextKernel is a proactive, semantic memory layer for LLM-based agents, designed to overcome the limitations of simple RAG and enable true long-term reasoning.**

[![Version](https://img.shields.io/badge/version-2.4-blue.svg)](https://github.com/your-repo/contextkernel)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Problem: LLMs Have Amnesia

Modern Large Language Models (LLMs) are incredibly powerful, but they have a fundamental flaw: they operate with the memory of a goldfish. Every interaction starts from a blank slate.

Retrieval-Augmented Generation (RAG) was a good first step, but it's a reactive band-aid, not a cure. It suffers from:

- **Context Blindness**: It dumps raw chunks of text into the prompt, often including irrelevant noise.
- **Inefficiency**: It wastes precious tokens and money by loading more context than necessary.
- **No Long-Term Learning**: It can't remember patterns, relationships, or context from previous sessions.

## The Solution: A Proactive Memory System

ContextKernel (CK) isn't just another vector database wrapper. It's a complete memory architecture that mimics a biological brain, with distinct layers for short-term, long-term, and raw sensory memory.

It works proactively, anticipating what context an agent needs and retrieving it with surgical precision. It also learns continuously, optimizing and storing new information to become smarter over time.

## Architecture Diagram (v2.4)

```mermaid
flowchart TD
    subgraph Inputs
        USER[User Input<br/>(Chat, Commands)]
        ENV[Environment Input<br/>(System Logs, APIs)]
    end

    subgraph CoreLogic [Orchestration & Reasoning Layer]
        CA[Context Agent<br/>(Proactive Orchestrator)]
        Retriever[LLM-1: Retriever<br/>(Searches & Reads Memory)]
        Listener[LLM-2: Listener<br/>(Summarizes, Saves & Updates Memory)]
    end

    subgraph MemorySystem [Unified Memory System]
        GraphDB[Graph DB<br/>(Central Index: Keywords, Links, Embeddings)]

        subgraph STM [STM - Short-Term Memory]
            direction LR
            S1[VDB-1] --- Sn[VDB-n]
        end

        subgraph LTM [LTM - Long-Term Memory]
            direction LR
            L1[VDB-1] --- Ln[VDB-n]
        end

        subgraph RawCache [Raw Logs - Immutable Archive]
            direction LR
            R1[Log 1] --- Rn[Log n]
        end
    end

    %% --- FLOWS ---
    USER --> CA
    ENV --> CA
    CA -->|1. Get relevant context| Retriever
    CA -->|4. Context missing, save this| Listener
    Retriever -->|2. Search Index| GraphDB
    GraphDB -.->|3a. Pointers| STM
    GraphDB -.->|3b. Pointers| LTM
    GraphDB -.->|3c. Pointers| RawCache
    Retriever -->|3d. Read selective data| STM
    Retriever -->|3d. Read selective data| LTM
    Retriever -->|3d. Read selective data| RawCache
    Listener -->|5. Save, Optimize & Update| STM
    Listener -->|5. Save, Optimize & Update| LTM
    Listener -->|5. Save, Optimize & Update| RawCache
    Listener -->|5. Save, Optimize & Update| GraphDB
```

## 🚀 Key Features

- ✅ **Proactive Context**: Automatically detects and injects relevant memory, even when not explicitly asked.
- 🧠 **Selective Recall**: Uses a Graph DB index to retrieve only the precise chunks of information needed, not entire documents.
- 📚 **Multi-Tier Memory**: Combines Short-Term (STM), Long-Term (LTM), and Raw Log databases for a layered, human-like memory system.
- 🕸️ **Unified Graph Spine**: All memory is interlinked and discoverable through a central knowledge graph, enabling complex, multi-hop reasoning.
- 🔄 **Continuous Learning**: An LLM-based "Listener" constantly processes new information, summarizes it, and integrates it into the memory system.
- 💰 **Token Optimization**: Dramatically reduces prompt sizes and API costs by eliminating irrelevant context.

## How It Works

ContextKernel operates on a sophisticated dual-loop system for reading and writing memory.

### 1. The Retrieval Flow (Reading Memory)

1. The Context Agent receives a task (e.g., a user query).
2. It dispatches the Retriever (LLM-1) to find context.
3. The Retriever queries the Graph DB first, asking "What information is related to this task?"
4. The Graph DB returns pointers to specific data chunks located in STM, LTM, or Raw Logs.
5. The Retriever performs a selective read, pulling only the specified chunks into the final context.

### 2. The Persistence Flow (Writing Memory)

1. The Context Agent identifies a "context gap"—new, valuable information that isn't in memory.
2. It dispatches the Listener (LLM-2) to process and save this information.
3. The Listener summarizes, compresses, and structures the new data.
4. It writes the data to the appropriate memory layer (e.g., a quick summary to STM, the raw data to the Raw Cache).
5. Crucially, it updates the Graph DB with new nodes and links, making the new memory discoverable for future queries.

## 📦 Installation

```bash
pip install contextkernel
```

## ⚡ Quick Start

Using ContextKernel is designed to be simple, hiding the architectural complexity behind a clean interface.

```python
import contextkernel as ck

# Initialize the kernel. This loads your configuration
# and connects to the underlying memory databases.
kernel = ck.Kernel()

# You don't need to manually fetch context. Just interact with the kernel.
# It will proactively find and inject relevant memory from past interactions,
# system logs, and documents.
prompt = "Why did our staging deployment fail last week? Was it the same database connection issue we saw in May?"

response = kernel.chat(prompt)

print(response)
# The response is generated with deep context, aware of both last week's
# staging logs and the incident report from May, without you having to
# manually load any documents.
```

## ContextKernel vs. Simple RAG

| Feature | Simple RAG | ContextKernel |
|---------|------------|---------------|
| Strategy | Reactive (Fetch on demand) | Proactive (Anticipates need) |
| Retrieval | Keyword/Vector search | Graph-indexed semantic search |
| Context | Dumps raw chunks | Injects precise, relevant data |
| Memory | Single, flat vector store | Multi-tier (STM, LTM, Raw) |
| Learning | Static | Continuous & self-optimizing |
| Efficiency | Low (High token waste) | High (Minimal token use) |
| Use Case | Simple Q&A | Complex, multi-session reasoning |

## 🗺️ Roadmap

- [ ] **More Database Adapters**: Support for additional Vector and Graph DBs.
- [ ] **Observability Suite**: A UI to visualize the context being retrieved and the memory graph.
- [ ] **Enterprise Security**: Enhanced roles, permissions, and data encryption.
- [ ] **On-Premise Deployment**: Packaged solutions for running CK in private clouds.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
