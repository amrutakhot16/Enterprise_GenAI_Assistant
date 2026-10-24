from langchain_core.chat_history import InMemoryChatMessageHistory

# One conversation per session_id (instead of one shared global history)
_histories: dict[str, InMemoryChatMessageHistory] = {}

# How many past messages to send to the LLM (keeps prompts small)
MAX_HISTORY_MESSAGES = 10


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _histories:
        _histories[session_id] = InMemoryChatMessageHistory()
    return _histories[session_id]


def recent_messages(session_id: str):
    return get_history(session_id).messages[-MAX_HISTORY_MESSAGES:]


def clear_history(session_id: str) -> None:
    _histories.pop(session_id, None)