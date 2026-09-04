from app.llm.model import get_llm
from app.llm.prompts import assistant_prompt
from app.rag.vectorstore import create_vector_store
from app.memory.chat_memory import chat_history

class ChatService:

    def __init__(self):

        # ==============================
        # Load LLM
        # ==============================

        print("Loading LLM...")

        self.llm = get_llm()

        # ==============================
        # Load vector database
        # ==============================

        print("Loading vector database...")

        self.vectorstore = create_vector_store()

        # ==============================
        # Create LLM chain
        # ==============================

        self.chain = assistant_prompt | self.llm


    # =================================
    # Extract sources from answer
    # =================================

    def extract_sources(self, answer):

        sources = []

        seen = set()

        if "Sources:" not in answer:
            return sources

        source_section = answer.split(
            "Sources:",
            1
        )[1]

        for line in source_section.splitlines():

            line = line.strip()

            if not line.startswith("-"):
                continue

            line = line.lstrip("-").strip()

            parts = line.rsplit(
                ", Page ",
                1
            )

            if len(parts) != 2:
                continue

            filename = parts[0].strip()

            try:
                page = int(parts[1].strip())
            except ValueError:
                continue

            key = (
                filename,
                page
            )

            if key not in seen:

                seen.add(key)

                sources.append(
                    {
                        "file": filename,
                        "page": page
                    }
                )

        return sources


    # =================================
    # Chat
    # =================================

    def chat(self, question):

        question = question.strip()

        # ==============================
        # Validate question
        # ==============================

        if not question:

            return {
                "answer": "Please provide a question.",
                "sources": []
            }


        # ==============================
        # Retrieve documents
        # ==============================

        results = (
            self.vectorstore
            .similarity_search_with_score(
                question,
                k=8
            )
        )


        # ==============================
        # No results
        # ==============================

        if not results:

            answer = (
                "I could not find this information "
                "in the provided documents."
            )

            return {
                "answer": answer,
                "sources": []
            }


        # ==============================
        # Build context
        # ==============================

        context_parts = []


        for doc, score in results:

            source = doc.metadata.get(
                "source",
                "unknown"
            )

            page = doc.metadata.get(
                "page",
                None
            )


            if page is not None:

                page_display = page + 1

            else:

                page_display = "unknown"


            context_parts.append(
                f"""
Source: {source}
Page: {page_display}

Content:
{doc.page_content}
"""
            )


        context = "\n\n".join(
            context_parts
        )


        # ==============================
        # Generate answer
        # ==============================

        response = self.chain.invoke(
            {
                "question": question,
                "context": context,
                "chat_history": chat_history.messages
            }
        )


        answer = response.content


        # ==============================
        # Determine sources
        # ==============================

        if (
            "I could not find this information "
            "in the provided documents."
            in answer
        ):

            sources = []

        else:

            sources = self.extract_sources(
                answer
            )


        # ==============================
        # Save conversation
        # ==============================

        chat_history.add_user_message(
            question
        )

        chat_history.add_ai_message(
            answer
        )


        # ==============================
        # Return response
        # ==============================

        return {
            "answer": answer,
            "sources": sources
        }