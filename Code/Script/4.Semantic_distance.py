
# Fourth Step: Semantic Distance

import numpy as np
import csv
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
import sys
import time
import logging

# --------------------------------------------------------
# Configure file paths
# --------------------------------------------------------

target_path = "/data/home/Zhuqian_He/originality" 
os.chdir(target_path)
sys.path.append(target_path)

# Constants
path_vectors = 'Data/Output/vectors/'
CHUNK_SIZE = 10000  # Adjust based on memory availability
OUTPUT_PATH = 'Data/Output/papers_cosine.csv'  # Adjust this path as needed
N_JOBS = -1  # Use all available cores

# --------------------------------------------------------
# Configure logging
# --------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('Log/4_Semantic_Distance.log'),
        logging.StreamHandler(sys.stdout)  
    ]
)

start_total = time.time()  
logging.info("=" * 50)
logging.info(f"Total process started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_total))}")
logging.info("=" * 50)


# Try to import cupy for GPU acceleration, fall back to numpy if not available
try:
    import cupy as xp
    logging.info("Running on GPU")
    USE_GPU = True
except ImportError:
    import numpy as xp
    logging.info("Running on CPU")
    USE_GPU = False 


# --------------------------------------------------------
# Define function
# --------------------------------------------------------
def load_vectors_for_year(year):
    """Load vectors for a specific year using efficient reading."""
    
    file_path = os.path.join(path_vectors, f"{year}_vectors.csv")
    
    if not os.path.exists(file_path):
        return None, None
    
    logging.info(f'Reading {year}...')
    # Load the entire CSV into a single numpy array
    try:
        data = xp.loadtxt(file_path, delimiter=',', dtype=xp.float32, skiprows=1)
    except Exception as e:
        logging.error(f"Failed to load vectors for year {year}: {str(e)}")
        return None, None  
    
    
    # Check if there is only one paper in the year
    if data.ndim == 1:
        papers_ids = xp.array([data[0].astype(xp.int64)])
        vectors = xp.array([data[1:]])
    else:
        # Slice the array to get the desired columns
        papers_ids = data[:, 0].astype(xp.int64)  # Assuming the first column is the PaperId
        vectors = data[:, 1:]  # Assuming the rest of the columns are the vectors

    return papers_ids, vectors

def cosine_similarity(vector_a, vector_b):
    """Simple cosine similarity function"""
    
    norm_a = xp.linalg.norm(vector_a) 
    norm_b = xp.linalg.norm(vector_b)
    
    dot_product = xp.dot(vector_a, vector_b)
    
    similarity = dot_product / (norm_a * norm_b)
    
    return similarity

def calculate_similarity_for_chunk(chunk, prior_data):
    """Calculate similarity for a chunk using matrix multiplication."""
    # Normalize the vectors
    chunk_norm = chunk / xp.linalg.norm(chunk, axis=1, keepdims=True)
    prior_data_norm = prior_data / xp.linalg.norm(prior_data, axis=1, keepdims=True)
    
    # Compute cosine similarities using matrix multiplication
    similarities = xp.dot(chunk_norm, prior_data_norm.T)
    
    avg_dists = xp.mean(similarities, axis=1)
    max_dists = xp.max(similarities, axis=1)
    
    return avg_dists, max_dists

def calculate_avg_max_similarity(current_data, prior_data):
    """Calculate average and max cosine similarities for chunks."""
    if USE_GPU:
        avg_similarities = []
        max_similarities = []
        for i in tqdm(range(0, len(current_data), CHUNK_SIZE)):
            chunk = current_data[i:i+CHUNK_SIZE]
            avg, max_ = calculate_similarity_for_chunk(chunk, prior_data)
            avg_similarities.append(avg)
            max_similarities.append(max_)
        return xp.concatenate(avg_similarities), xp.concatenate(max_similarities)
    else:
        results = Parallel(n_jobs=N_JOBS)(
            delayed(calculate_similarity_for_chunk)(current_data[i:i+CHUNK_SIZE], prior_data)
            for i in tqdm(range(0, len(current_data), CHUNK_SIZE))
        )
        avg_similarities = xp.concatenate([res[0] for res in results])
        max_similarities = xp.concatenate([res[1] for res in results])
        return avg_similarities, max_similarities

def initialize_output_file():
    """Initialize the output CSV file with headers."""
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['PaperID', 'cosine_max', 'cosine_avg', 'semantic_distance'])

def save_to_csv(papers_ids, avg_similarities, max_similarities, semantic_distance):
    """Append results to CSV."""
    if 'cupy' in str(type(papers_ids)):
        papers_ids = papers_ids.get()
    if 'cupy' in str(type(avg_similarities)):
        avg_similarities = avg_similarities.get()
    if 'cupy' in str(type(max_similarities)):
        max_similarities = max_similarities.get()
    if 'cupy' in str(type(semantic_distance)):
        semantic_distance = semantic_distance.get()
    with open(OUTPUT_PATH, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for paper_id, avg_sim, max_sim, sem_dist in zip(papers_ids, avg_similarities, max_similarities, semantic_distance):
            writer.writerow([paper_id, max_sim, avg_sim, sem_dist])
            
start_year = 2019
end_year = 2024

rolling_data = []
years = range(start_year, end_year + 1) # +1 to include the last year

# Initialize the output CSV file
initialize_output_file()

for year in tqdm(years):
    papers_ids, current_year_data = load_vectors_for_year(year)
    
    if current_year_data is None:
        continue

    # Add current year data to rolling data
    rolling_data.append((year, current_year_data))
    
    # Remove data that is more than 5 years old
    rolling_data = [(y, data) for y, data in rolling_data if year - y < 6]

    # If there's not enough prior data, skip the calculations for this year
    if len(rolling_data) < 6:
        continue

    # Combine prior years data
    prior_data = xp.vstack([data for y, data in rolling_data if y != year])
    
    logging.info('Calculating similarities for %d...'%(year))
    # Calculate cosine similarities
    avg_year_similarities, max_year_similarities = calculate_avg_max_similarity(current_year_data, prior_data)
    
    semantic_distance = 1 - max_year_similarities

    # Save results to CSV
    save_to_csv(papers_ids, avg_year_similarities, max_year_similarities, semantic_distance)
    
end_total = time.time()
total_duration = end_total - start_total
logging.info("=" * 50)
logging.info(f"Total process finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_total))}")
logging.info(f"Total duration: {total_duration:.2f} seconds ({total_duration / 60:.1f} minutes)")
logging.info("=" * 50)
