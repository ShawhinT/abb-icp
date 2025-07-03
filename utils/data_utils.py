import pandas as pd
import re

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

def extract_cohort_number(filename):
    """
    Extract the cohort number from an AI Builders Bootcamp filename.
    
    This function uses regex to find and extract the cohort number from filenames
    that follow the pattern 'AI Builders Bootcamp_{number}_...'.
    
    Args:
        filename (str): The filename or filepath containing the cohort number.
                       Expected format: 'AI Builders Bootcamp_{cohort_number}_...'
    
    Returns:
        str or None: The cohort number as a string if found, None if no match.
    """
    pattern = r'AI Builders Bootcamp_(\d+)_'
    match = re.search(pattern, filename)
    
    if match:
        cohort_number = match.group(1)
        
        return cohort_number
    
def df_to_markdown(df, filepath, title=None):
    """
    Write a pandas DataFrame to a markdown table and save it to a file.

    Args:
        df (pd.DataFrame): The DataFrame to convert.
        filepath (str): The path to the file where the markdown will be saved.
        title (str, optional): If provided, will be written as a markdown header at the top.
    """
    markdown = df.to_markdown(index=False)
    with open(filepath, "w") as f:
        if title:
            f.write(f"### {title}\n\n")
        f.write(markdown) 