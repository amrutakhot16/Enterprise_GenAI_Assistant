import os
import warnings

from langchain_core.output_parsers import StrOutputParser

from app.llm.model import get_llm
from app.llm.prompts import (
    NOT_FOUND_MESSAGE,
    assistant_prompt,
    rewrite_prompt,
)
from app.memory.chat_memory import get_history, recent_messages
from app.rag.vectorstore import create_vector_store

warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")

TOP_K = 6

# Relevance is 0..1 (higher = better). Chunks below this are ignored.
# Tune it: print the scores for good and bad questions and pick a cut-off.
MIN_RELEVANCE = 0.3


class ChatService:

    def __init__(self):
        print("Loading LLM...")
        self.llm = get_llm()

        print("Loading vector database...")
        self.vectorstore = create_vector_store()

        self.answer_chain = assistant_prompt | self.llm | StrOutputParser()
        self.rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()

    # ---------------------------------
    # Turn "how much does it cost?" into a standalone question
    # ---------------------------------

    def standalone_question(self, question, history):
        if not history:
            return question

        rewritten = self.rewrite_chain.invoke(
            {"question": question, "chat_history": history}
        ).strip()

        return rewritten or question

    # ---------------------------------
    # Retrieve relevant chunks
    # ---------------------------------

    def retrieve(self, query):
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=TOP_K,
        )

        print(f"\nSearch: {query}")
        for doc, score in results:
            file = os.path.basename(doc.metadata.get("source", ""))
            print(f"  {score:.3f}  {file} page {doc.metadata.get('page')}")

        return [
            (doc, score)
            for doc, score in results
            if score >= MIN_RELEVANCE
        ]

    # ---------------------------------
    # Build the context and the sources list from metadata
    # ---------------------------------

    @staticmethod
    def build_context(results):
        context_parts = []
        sources = {}

        for doc, score in results:
            file = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page")
            page_display = page + 1 if page is not None else None

            context_parts.append(
                f"Source: {file}, Page {page_display}\n{doc.page_content}"
            )

            key = (file, page_display)
            if key not in sources or score > sources[key]["score"]:
                sources[key] = {
                    "file": file,
                    "page": page_display,
                    "score": round(score, 3),
                }

        sorted_sources = sorted(
            sources.values(),
            key=lambda s: s["score"],
            reverse=True,
        )

        return "\n\n---\n\n".join(context_parts), sorted_sources

    # ---------------------------------
    # Streaming chat: yields events one by one
    #   {"type": "token", "text": "..."}   a piece of the answer
    #   {"type": "done", "sources": [...]} sent once at the end
    # ---------------------------------

    def stream_chat(self, question, session_id):
        question = question.strip()

        if not question:
            yield {"type": "token", "text": "Please provide a question."}
            yield {"type": "done", "sources": [], "search_query": ""}
            return

        history = recent_messages(session_id)

        search_query = self.standalone_question(question, history)
        results = self.retrieve(search_query)

        answer_parts = []
        sources = []

        if not results:
            answer_parts.append(NOT_FOUND_MESSAGE)
            yield {"type": "token", "text": NOT_FOUND_MESSAGE}
        else:
            context, sources = self.build_context(results)

            for piece in self.answer_chain.stream(
                {
                    "question": question,
                    "context": context,
                    "chat_history": history,
                }
            ):
                if piece:
                    answer_parts.append(piece)
                    yield {"type": "token", "text": piece}

        answer = "".join(answer_parts).strip()

        if NOT_FOUND_MESSAGE in answer:
            sources = []

        session = get_history(session_id)
        session.add_user_message(question)
        session.add_ai_message(answer)

        yield {
            "type": "done",
            "sources": sources,
            "search_query": search_query,
        }

    # ---------------------------------
    # Non-streaming chat (same logic, collected into one response)
    # ---------------------------------

    def chat(self, question, session_id):
        answer_parts = []
        final = {}

        for event in self.stream_chat(question, session_id):
            if event["type"] == "token":
                answer_parts.append(event["text"])
            else:
                final = event

        return {
            "answer": "".join(answer_parts).strip(),
            "sources": final.get("sources", []),
            "search_query": final.get("search_query", ""),
        }