from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

NOT_FOUND_MESSAGE = (
    "I could not find this information in the provided documents."
)

# ---------------------------------
# Answer prompt
# ---------------------------------

assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a document question answering assistant.

Rules:
1. Answer ONLY from the context below.
2. Do not use general knowledge and do not invent facts.
3. Do not repeat the context or mention the word "context".
4. Do not list sources; they are added automatically.
5. If the answer is not in the context, reply exactly:
"""
            + NOT_FOUND_MESSAGE
            + """

Context:
{context}""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

# ---------------------------------
# Follow-up question rewriting
# ---------------------------------

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation so far and a follow-up question, "
            "rewrite the follow-up as a standalone question that can be "
            "understood without the conversation. Do NOT answer it. "
            "If it is already standalone, return it unchanged. "
            "Return only the question.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)