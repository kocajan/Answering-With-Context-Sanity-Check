import ollama

from google import genai


def make_search_query_cloud(question: str, system_prompt: str, client: genai.Client) -> str:
    """
    Generate a search query using the cloud-based Gemini model.

    Args:
        question (str): The user's question.
        system_prompt (str): The system prompt for query generation.
        client: The cloud API client.

    Returns:
        str: The generated search query.
    """
    prompt = system_prompt.format(question=question)
    r = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return r.text.strip()

def make_search_query_local(question: str, system_instruction: str) -> str:
    """
    Generate a search query using the local Ollama model.

    Args:
        question (str): The user's question.
        system_instruction (str): The system instruction for query generation.

    Returns:
        str: The generated search query.
    """
    prompt = system_instruction.format(question=question)
    r = ollama.generate(
        model="llama3:8b",
        prompt=prompt
    )
    return r["response"].strip()

def make_search_query(question: str, system_instruction: str, client: genai.Client = None) -> str:
    """
    Generate a search query using the specified model.

    Args:
        question (str): The user's question.
        system_instruction (str): The system instruction for query generation.
        client: The cloud API client (if using cloud generation).

    Returns:
        str: The generated search query.
    """
    if client is not None:
        return make_search_query_cloud(question, system_instruction, client)
    return make_search_query_local(question, system_instruction)
