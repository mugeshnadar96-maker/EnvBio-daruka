# LangChain Adapter Notes

`conversation_agent.py` implements the same *behavioral contract* the spec
asks for from LangChain, without importing the `langchain` metapackage
(unavailable to install fully within this sandbox's disk/network budget).
In an unrestricted environment, swap these in directly:

## Memory
Replace the `deque(maxlen=5)` + `SiteState` accumulation in `EnvAgent`
with:

```python
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=5, return_messages=True)
```

Keep the `SiteState` merge logic as-is (it's not a chat-history concern,
it's structured slot extraction) — you'd store `SiteState` as a custom
field alongside the LangChain memory object, or serialize it into memory's
`memory_variables`.

## Retriever
Replace `Retriever.retrieve()` (raw `chromadb` client calls) with:

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

vectordb = Chroma(
    collection_name="environmental_knowledge",
    persist_directory="./env_knowledge_db",
    embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
docs = retriever.get_relevant_documents(query)
```

## Agent wrapper
The intent classifier + rule engine + retriever fusion in `handle_turn()`
maps directly onto a LangChain `AgentExecutor` with three tools
(`retrieve_evidence`, `run_reasoning_rules`, `ask_clarifying_question`) if
you want the LLM itself to choose the control flow instead of the
deterministic Python router used here. The deterministic router was kept
intentionally in this build because the spec requires **hardcoded**
reasoning rules with guaranteed field coverage (all 6 mandatory output
fields) — a tool-calling LLM agent risks dropping fields non-deterministically
unless heavily constrained, which is a worse fit for an evaluation with a
strict output schema.
