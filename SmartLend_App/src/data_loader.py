import pandas as pd
import numpy as np
import time

def load_data_chunked(file_path: str, chunk_size: int = 500000) -> pd.DataFrame:
    """
    Loads a large CSV dataset in chunks to avoid running out of RAM (memory).
    It downcasts the data types of each chunk dynamically to reduce the memory footprint.
    
    Parameters:
    - file_path: The local system path to the CSV file (e.g. loan.csv).
    - chunk_size: Number of rows to read at a time.
    """
    print(f"[DataLoader] Loading data from {file_path} in chunks of {chunk_size}...")
    start_time = time.time()
    
    chunks = []
    
    # We iterate over the CSV file in chunks instead of loading all 2.2M rows at once
    chunk_iter = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)
    
    for i, chunk in enumerate(chunk_iter):
        # Downcast columns to smaller types to save RAM
        optimized_chunk = optimize_datatypes(chunk)
        chunks.append(optimized_chunk)
        
        print(f"[DataLoader] Processed chunk {i+1} (approx { (i+1) * chunk_size } rows)...")
        
    # Combine all optimized chunks into a single consolidated DataFrame
    df = pd.concat(chunks, ignore_index=True)
    
    elapsed = time.time() - start_time
    print(f"[DataLoader] Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns in {elapsed:.1f} seconds!")
    print(f"[DataLoader] Total optimized DataFrame Memory: {df.memory_usage().sum() / (1024*1024):.1f} MB")
    
    return df

def optimize_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scans a DataFrame's columns and downcasts integers and floats to 
    smaller byte sizes, and objects to category types, to minimize RAM usage.
    """
    for col in df.columns:
        col_type = df[col].dtype
        
        # If the column is numerical (integers or decimals)
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # If the column contains integers
            if str(col_type).startswith('int'):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            # If the column contains decimals (floats)
            else:
                # Float32 uses half the memory of Float64 and has plenty of precision
                df[col] = df[col].astype(np.float32)
        # If the column contains text/categorical strings
        else:
            df[col] = df[col].astype('category')
            
    return df
