from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np

# setup api client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def compute_openai_embedding(text, column_name: str, model_name: str = 'text-embedding-3-small', dimensions: int = 1536):
    """
    Compute an embedding for a given text using OpenAI's embeddings API.
    
    Args:
        text (str or list-like): The text to compute an embedding for
        column_name (str): The name of the column to compute an embedding for
        model_name (str): OpenAI model to use (default: 'text-embedding-3-small')
    
    Returns:
        embeddings: numpy array of shape (N, 1536)
    """
    response = client.embeddings.create(input = f"{column_name}: " + text, model=model_name, dimensions=dimensions)
    embedding_list = [embedding_object.embedding for embedding_object in response.data]
    return np.array(embedding_list)

def send_openai_request(instructions: str, user_input: str, model_name: str = 'gpt-4.1', temperature: float = 1.0):
    """
    Send a request to OpenAI's responses API.

    Args:
        instructions (str): The instructions for the response
        user_input (str): The input for the response
        model_name (str): OpenAI model to use (default: 'gpt-4.1')
        temperature (float): Temperature for the response (default: 1.0)

    Returns:
        str: Response from OpenAI's responses API
    """
    response = client.responses.create(
        model=model_name,
        instructions=instructions,
        input=user_input,
        temperature=temperature
    )
    return response.output_text