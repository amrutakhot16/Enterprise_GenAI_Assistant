import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def get_llm():

    api_key = os.getenv("GROQ_API_KEY")

    model = os.getenv(
        "GROQ_MODEL",
        "groq/compound-mini"
    )

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing from .env"
        )

    return ChatGroq(
        model=model,
        temperature=0,
        api_key=api_key,
    )


if __name__ == "__main__":

    llm = get_llm()

    response = llm.invoke(
        "Hello! Test connection."
    )
    print(response.content)