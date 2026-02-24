# backend/rag_graph.py
from __future__ import annotations

from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from rag_store import get_vector_store

class RAGState(TypedDict):
    question: str
    docs: List[Document]
    answer: str

def build_rag_graph():
    vs = get_vector_store()
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="gpt-5-nano",
        temperature=0.2,
        # streaming is handled via graph streaming modes; model supports token streaming :contentReference[oaicite:10]{index=10}
    )

    def retrieve(state: RAGState) -> dict:
        docs = retriever.invoke(state["question"])
        return {"docs": docs}

    def generate(state: RAGState) -> dict:
        context = "\n\n".join(
            f"[source={d.metadata.get('source','')}] {d.page_content}"
            for d in state["docs"]
        )
        prompt = (
            "You are a helpful assistant. Answer using ONLY the context.\n\n"
            f"Question:\n{state['question']}\n\n"
            f"Context:\n{context}\n\n"
            "If the answer is not in the context, say: 'Not found in the provided documents.'"
        )

        msg = llm.invoke(prompt)
        text = msg.content if isinstance(msg, AIMessage) else str(msg)
        return {"answer": text}

    graph = (
        StateGraph(RAGState)
        .add_node("retrieve", retrieve)
        .add_node("generate", generate)
        .add_edge(START, "retrieve")
        .add_edge("retrieve", "generate")
        .add_edge("generate", END)
        .compile()
    )
    return graph