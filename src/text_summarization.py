import ollama

from google import genai


def summarize_text_local(text: str, system_prompt: str) -> str:
    """
    Summarize text using the local Ollama model.

    Args:
        text (str): The text to summarize.
        system_prompt (str): The system prompt for summarization.

    Returns:
        str: The summarized text.
    """
    prompt = system_prompt.format(text=text)
    r = ollama.generate(model="llama3:8b", prompt=prompt)
    return r["response"].strip()

def summarize_text_cloud(text: str, system_prompt: str, client: genai.Client) -> str:
    """
    Summarize text using the cloud-based model.

    Args:
        text (str): The text to summarize.
        system_prompt (str): The system prompt for summarization.
        client: The cloud API client.

    Returns:
        str: The summarized text.
    """
    prompt = system_prompt.format(text=text)
    r = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return r.text.strip()


def summarize_pages(pages: dict, system_prompt: str, client: genai.Client = None, text_length_limit: int = 5000) -> dict:
    """
    Summarize multiple pages of text locally or using the cloud if a client is provided.

    Args:
        pages (dict): A dictionary mapping URLs to their extracted text.
        system_prompt (str): The system prompt for summarization.
        client: The cloud API client (if using cloud summarization).
        text_length_limit (int): The maximum length of text to summarize.

    Returns:
        dict: A dictionary mapping URLs to their summarized text.
    """
    summaries = {}
    for url, text in pages.items():
        short_text = text[:text_length_limit]
        if client is not None:
            summaries[url] = summarize_text_cloud(short_text, system_prompt, client)
        else:
            summaries[url] = summarize_text_local(short_text, system_prompt)
    return summaries
