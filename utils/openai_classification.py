from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import pandas as pd

# setup api client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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