from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# setup api client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_clean_data(data_name):
    """
    Load a cleaned CSV dataset from the data/2-clean directory.
    
    Args:
        data_name (str): Name of the dataset file (without .csv extension)
        
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    filename = f'data/2-clean/{data_name}.csv'
    return pd.read_csv(filename)

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

# use openai responses api to classify company name as solo, smb, or enterprise
def classify_company_size(company_name: str, model_name: str = 'gpt-4.1-mini'):
    """
    Classify a company's size using OpenAI's responses API.
    
    Args:
        company_name (str): The name of the company to classify
        model_name (str): OpenAI model to use (default: 'gpt-4.1-mini')
    
    Returns:
        CompanySize: Object with reasoning and category (solo/smb/enterprise/unknown)
    """
    class CompanySize(BaseModel):
        reasoning: str
        category: str

    instructions = open('prompts/company_classifier.txt', 'r').read()

    if company_name == 'Unknown company':
        company_size = CompanySize(
            reasoning="Unknown company", 
            category="unknown", 
        )
        return company_size

    print(f"Classifying company size for: {company_name}...")
    response = client.responses.parse(
        model=model_name,
        instructions=instructions,
        input=f"Company name: {company_name}",
        tools=[{"type": "web_search_preview"}],
        text_format=CompanySize,
        temperature=0.0
    )
    return response.output_parsed

def classify_all_company_sizes(df: pd.DataFrame, column_name: str, model_name: str = 'gpt-4.1-mini'):
    """
    Classify all company sizes in a given column using OpenAI's responses API.
    First checks if existing classifications exist in CSV file to avoid redundant LLM calls.
    
    Args:
        df (pd.DataFrame): The dataframe containing the company names
        column_name (str): The name of the column containing the company names
        model_name (str): OpenAI model to use (default: 'gpt-4.1-mini')
    
    Returns:
        pd.DataFrame: DataFrame with company classifications
    """
    csv_path = 'data/2-clean/company_sizes.csv'
    
    # Load existing data if available
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        existing_companies = set(existing_df['company_name'])
        print(f"Loaded {len(existing_companies)} existing company classifications")
    else:
        existing_df = pd.DataFrame(columns=['company_name', 'company_size_category', 'company_size_reasoning'])
        existing_companies = set()
    
    # Find companies that need classification
    input_companies = df[column_name].tolist()
    missing_companies = [name for name in input_companies if name not in existing_companies]
    
    # Classify missing companies
    if missing_companies:
        print(f"Classifying {len(missing_companies)} new companies...")
        new_rows = []
        for company_name in missing_companies:
            response = classify_company_size(company_name, model_name)
            new_rows.append({
                'company_name': company_name,
                'company_size_category': response.category,
                'company_size_reasoning': response.reasoning
            })
        
        # Append and save
        new_df = pd.DataFrame(new_rows)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_df.to_csv(csv_path, index=False)
    else:
        updated_df = existing_df
    
    # Return results filtered to input companies in correct order
    return updated_df[updated_df['company_name'].isin(input_companies)].set_index('company_name').reindex(input_companies).reset_index()

def classify_job_category(job_title: str, company_size: str, model_name: str = 'gpt-4.1-nano'):
    """
    Classify a job title into one of the following categories:
    - ic
    - manager
    - leader
    - entrepreneur
    - student
    
    Args:
        job_title (str): The job title to classify
        company_size (str): The size of the company (solo/smb/enterprise)
        model_name (str): OpenAI model to use (default: 'gpt-4.1-mini')
    
    Returns:
        JobCategory: Object with reasoning and category (data engineer/data analyst/data scientist/machine learning engineer/ai engineer/other)
    """
    class JobCategory(BaseModel):
        reasoning: str
        category: str

    instructions = open('prompts/job_classifier.txt', 'r').read()

    if job_title == 'Unknown job title':
        job_category = JobCategory(
            reasoning = "Unknown job title", 
            category = "unknown", 
        )
        return job_category
    
    if company_size == 'solo':
        job_category = JobCategory(
            reasoning = "Solo company", 
            category = "entrepreneur", 
        )
        return job_category

    print(f"Classifying job title for: {job_title} at {company_size}...")
    response = client.responses.parse(
        model=model_name,
        instructions=instructions,
        input=f"Job title: {job_title}",
        text_format=JobCategory,
        temperature=0.0
    )
    return response.output_parsed

def classify_all_job_categories(df: pd.DataFrame, job_title_column_name: str, company_size_column_name: str, model_name: str = 'gpt-4.1-nano'):
    """
    Classify all job categories in a dataframe using OpenAI's responses API.
    
    Args:
        df (pd.DataFrame): The dataframe containing job titles and company sizes
        job_title_column_name (str): The name of the column containing job titles
        company_size_column_name (str): The name of the column containing company sizes
        model_name (str): OpenAI model to use (default: 'gpt-4.1-nano')
    
    Returns:
        dict: Dictionary containing two lists:
            - 'job_category': List of classified job categories (ic/manager/leader/entrepreneur/student/unknown)
            - 'job_category_reasoning': List of reasoning for each classification
    """
    job_category_list = []
    job_category_reasoning_list = []

    for job_title, company_size in zip(df[job_title_column_name], df[company_size_column_name]):
        response = classify_job_category(job_title, company_size, model_name)
        job_category_list.append(response.category)
        job_category_reasoning_list.append(response.reasoning)
    return {
        'job_category': job_category_list,
        'job_category_reasoning': job_category_reasoning_list,
    }

def compute_openai_embedding(text, column_name: str, model_name: str = 'text-embedding-3-small'):
    """
    Compute an embedding for a given text using OpenAI's embeddings API.
    
    Args:
        text (str or list-like): The text to compute an embedding for
        column_name (str): The name of the column to compute an embedding for
        model_name (str): OpenAI model to use (default: 'text-embedding-3-small')
    
    Returns:
        embeddings: numpy array of shape (N, 1536)
    """
    response = client.embeddings.create(input = f"{column_name}: " + text, model=model_name)
    embedding_list = [embedding_object.embedding for embedding_object in response.data]
    return np.array(embedding_list)

def compute_principal_components(embeddings: np.array, n_components: int = 5):
    """
    Compute principal components for a given set of embeddings.
    
    Args:
        embeddings (np.array): The embeddings to compute principal components for
        n_components (int): The number of principal components to compute (default: 5)
    
    Returns:
        principal_components: numpy array of shape (N, n_components) with values in [-1, 1]
    """
    # Compute PCA
    pca = PCA(n_components=n_components, whiten=True)
    principal_components = pca.fit_transform(embeddings)
    
    # Scale each column to [-1, 1] range
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_components = scaler.fit_transform(principal_components)
    
    return scaled_components

def greater_than_or_equal_x(df, col_name, x):
    """
    Insert a new boolean column indicating rows with values greater than or equal to X.
    
    Args:
        df (pandas.DataFrame): The DataFrame to modify
        col_name (str): Name of the column to compare against
        x (int/float): The threshold value
        
    Returns:
        None: Modifies the DataFrame in place
    """
    # Create boolean series
    boolean_values = df[col_name] >= x
    
    # Find position of the original column
    col_position = df.columns.get_loc(col_name)
    
    # Create new column name
    new_col_name = f'{col_name}_gte_{x}'
    
    # Insert the new boolean column right after the original column
    df.insert(
        col_position + 1,  # Position to insert (after original column)
        new_col_name,      # Column name
        boolean_values     # Column values
    )

def create_category_boolean_columns(df, category_column, placement_column=None):
    """
    Create boolean columns for each category in a categorical column and insert them after a specified column.
    
    Args:
        df: DataFrame to modify in-place
        category_column: Name of the column containing categories
        placement_column: Name of the column after which to insert the new boolean columns
    
    Returns:
        n/a (Modifies df in-place)
    """
    if placement_column is None:
        placement_column = category_column

    # Find the position of the placement column
    placement_position = df.columns.get_loc(placement_column)
    
    # Get unique categories (excluding NaN values)
    unique_categories = np.sort(df[category_column].dropna().unique())
    
    # Create the boolean columns
    boolean_columns = {}
    for category in unique_categories:
        category_clean = str(category).lower().replace(' ', '_')
        col_name = f'{category_column}_{category_clean}'
        boolean_columns[col_name] = df[category_column] == category
    
    # Insert the boolean columns after the placement column
    for i, (col_name, col_values) in enumerate(boolean_columns.items()):
        df.insert(
            placement_position + 1 + i,  # Position after placement column + previous boolean columns
            col_name,                    # Column name
            col_values.astype(int)       # Column values as integers (0/1)
        )