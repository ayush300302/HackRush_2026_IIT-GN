import pandas as pd
import os
import time
import numpy as np

def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """ Iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32) # float16 has issues with some pandas ops, using float32
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # We can convert object columns to category to save memory
            df[col] = df[col].astype('category')
            
    return df

def load_data_chunked(file_path: str, chunk_size: int = 100000) -> pd.DataFrame:
    """
    Loads a large CSV file using Pandas chunking to reduce memory overhead.
    """
    print(f"Loading data from: {file_path} in chunks of {chunk_size}...")
    start_time = time.time()
    
    chunks = []
    try:
        # We iterate over the file in chunks
        chunk_iter = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)
        for i, chunk in enumerate(chunk_iter):
            # Downcast types in each chunk before appending to save memory
            optimized_chunk = reduce_mem_usage(chunk)
            chunks.append(optimized_chunk)
            if (i+1) % 5 == 0:
                print(f"Processed { (i+1) * chunk_size } rows...")
                
        # Concatenate all optimized chunks
        df = pd.concat(chunks, ignore_index=True)
        
        end_time = time.time()
        print(f"Data loaded and optimized successfully in {end_time - start_time:.2f} seconds.")
        print(f"Dataset Shape: {df.shape}")
        
        mem_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"Optimized Memory Usage: {mem_usage:.2f} MB")
        return df
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None

if __name__ == "__main__":
    dataset_path = r"C:\Users\siddp\Downloads\Dataset for default loan prediction\loan.csv"
    if os.path.exists(dataset_path):
        df = load_data_chunked(dataset_path)
    else:
        print(f"Dataset not found at {dataset_path}")
