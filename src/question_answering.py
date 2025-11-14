from google import genai

from .text_summarization import summarize_pages
from .search_query_generation import make_search_query
from .answering_with_context import combine_and_answer
from .google_page_extraction import google_search, extract_from_urls


def get_answer_for_question(config: dict, question: str, client: genai.Client = None) -> str:
    """
    Get an answer for a single question using the provided configuration.

    Args:
        config (dict): The configuration dictionary.
        question (str): The user's question.
        client: The cloud API client (if using cloud generation).

    Returns:
        str: The answer to the question.
    """
    # Check if client is provided for cloud usage
    version = "cloud" if client is not None else "local"

    # Generate search query
    print("     - Generating search query...")
    search_query = make_search_query(
        question,
        config['system_prompts']['query_generation'][version],
        client
    )

    # Perform Google search
    print("     - Performing Google search...")
    urls = google_search(
        search_query,
        api_key=config['api_keys']['google_search_api_key'],
        cse_id=config['custom_search_engine_id'],
        n=config['search_results']['num_results'],
        timeout=config['search_results']['timeout']
    )

    # Fetch and extract text from URLs
    print("     - Fetching and extracting text from URLs...")
    pages = extract_from_urls(urls)

    # Summarize pages
    print("     - Summarizing extracted texts...")
    summaries = summarize_pages(
        pages,
        config['system_prompts']['summarization'][version],
        client,
        text_length_limit=config['summarization']['text_length_limit']
    )

    # Combine summaries and answer the question
    print("     - Generating final answer...")
    answer = combine_and_answer(
        question,
        summaries,
        config['system_prompts']['answer_generation'][version],
        client
    )

    return answer

def get_answers_for_questions(config: dict, questions: list) -> dict:
    """
    Get answers for a list of questions using the provided configuration.

    Args:
        config (dict): The configuration dictionary.
        questions (list): A list of questions.

    Returns:
        dict: A dictionary mapping questions to their answers.
    """
    # Get the client if using cloud
    if config['use_cloud']:
        client = genai.Client(api_key=config['api_keys']['gemini_api_key'])
    else:
        client = None

    # Iterate over questions and get answers
    answers = {}
    print("Processing questions on the", "cloud..." if client else "local machine...")
    for question in questions:
        print(f"- Processing question: {question}")
        answers[question] = get_answer_for_question(config, question, client)

    return answers
