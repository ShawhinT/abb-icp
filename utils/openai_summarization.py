from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd

# setup api client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_response(response: str, model_name: str = 'gpt-4.1-nano'):
    """
    Summarize a response using OpenAI's responses API.

    Args:
        response (str): The response to summarize
        model_name (str): OpenAI model to use (default: 'gpt-4.1-nano')

    Returns:
        str: The summarized response
    """
    if response == 'No response provided':
        return 'No response provided'

    instructions = open('prompts/response_summarizer.txt', 'r').read()

    print(f"Summarizing response for: {response[:100]}...")
    response = client.responses.create(
        model=model_name,
        instructions=instructions,
        input=response,
        temperature=0.0
    )
    return response.output_text

def summarize_all_responses(df: pd.DataFrame, column_name: str, model_name: str = 'gpt-4.1-nano'):
    """
    Summarize all responses in a given column using OpenAI's responses API.

    Args:
        df (pd.DataFrame): The dataframe containing the responses
        column_name (str): The name of the column containing the responses
        model_name (str): OpenAI model to use (default: 'gpt-4.1-nano')

    Returns:
        list: List of summarized responses
    """
    summarized_responses = []
    for response in df[column_name]:
        summarized_responses.append(summarize_response(response, model_name))
    return {
        f'{column_name}_summarized': summarized_responses,
    }