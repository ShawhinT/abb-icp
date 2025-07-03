import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

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