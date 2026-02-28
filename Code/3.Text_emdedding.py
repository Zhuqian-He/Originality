
# Third Step: Text Emdedding

import os
import csv
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import sys
import logging
import time
import itertools

# --------------------------------------------------------
# Configure file paths
# --------------------------------------------------------

target_path = "/data/home/Zhuqian_He/originality" 
os.chdir(target_path)
sys.path.append(target_path)


## Increase the max size of a line reading, otherwise an error is raised
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
        
# Constants
PATH_OUTPUT = 'Data/Output'
PATH_INPUT = 'Data/Raw/filter_paper.csv'
STORAGE = 'csv'
CHUNK_SIZE = 50
TOTAL_PAPERS = None

# --------------------------------------------------------
# Configure logging
# --------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('Log/3_Text_embedding.log'),
        logging.StreamHandler(sys.stdout)  
    ]
)

start_total = time.time()  
logging.info("=" * 50)
logging.info(f"Total process started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_total))}")
logging.info("=" * 50)


# --------------------------------------------------------
# Define function
# --------------------------------------------------------
# Move the model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_name = 'specter'):
    
    if 'specter' in model_name.lower():
        model = 'allenai/specter'
    elif 'scibert' in model_name.lower():
        model = 'allenai/scibert_scivocab_uncased'
    else:
        model = model_name
        
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    
    return tokenizer, model

def get_embedding(texts, tokenizer, model, max_length=512):
    inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1) 
    return embeddings

# Check if paths exist
if not os.path.exists(PATH_OUTPUT) or not os.path.exists(PATH_INPUT):
    raise Exception("Input or output path does not exist.")

# Load the embedding model
logging.info('Loading the embedding model...')
tokenizer, model = load_model()

model.to(device)
logging.info(f"Using {device.upper()}.")

# Count the number of papers
logging.info('Get the number of papers to process...')
with open(PATH_INPUT, 'r', encoding='utf-8') as file:
    line_count = sum(1 for line in file)
TOTAL_PAPERS = line_count - 1  # Subtract 1 for the header


def get_last_processed_index(path_output):
    total_processed = 0
    vectors_path = os.path.join(path_output, 'vectors')
    
    if not os.path.exists(vectors_path):
        return total_processed
    
    for file in os.listdir(vectors_path):
        if file.endswith('.csv') or file.endswith('.npy'):
            file_path = os.path.join(vectors_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                total_processed += sum(1 for line in f) - 1  # Subtract 1 to exclude the header row
    
    return total_processed

def save_vectors(vectors, year, storage, path_output):
    vectors_path = os.path.join(path_output, 'vectors')
    os.makedirs(vectors_path, exist_ok=True)  # Ensure the directory exists
    
    file_path = os.path.join(vectors_path, f'{year}_vectors')
    
    if storage == 'csv':
        file_path += '.csv'
        mode = 'a' if os.path.exists(file_path) else 'w'
        with open(file_path, mode, encoding='utf-8', newline='') as writer:
            csv_writer = csv.writer(writer, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if mode == 'w':
                logging.info(f'Creating new file for year {year}...')
                csv_writer.writerow(["PaperID"] + [f"{i}" for i in range(len(vectors[0]) - 1)])  # Adjusted header format
            csv_writer.writerows(vectors)
    elif storage == 'numpy':
        file_path += '.npy'
        vectors = np.array([vec[1:] for vec in vectors])  # Exclude PaperID for numpy storage
        if os.path.exists(file_path):
            existing_vectors = np.load(file_path, allow_pickle=True)
            vectors = np.vstack((existing_vectors, vectors))
        np.save(file_path, vectors)
    else:
        raise ValueError("Unsupported storage format. Use 'csv' or 'numpy'.")

def process_papers(start_index):
    with open(PATH_INPUT, 'r', encoding='utf-8') as reader:
        csv_reader = csv.reader(reader, delimiter=',', quotechar='"')

        # Skip headers and already processed papers
        logging.info('Already done papers...')
        csv_reader = itertools.islice(csv_reader, start_index + 1, None)

        for chunk_start in tqdm(range(start_index, TOTAL_PAPERS, CHUNK_SIZE)):
            chunk_data = [line_csv for _, line_csv in zip(range(CHUNK_SIZE), csv_reader)]
            
            # Group by year
            papers_by_year = {}
            for data in chunk_data:
                year = int(data[1].split('-')[0])
                if year not in papers_by_year:
                    papers_by_year[year] = []
                papers_by_year[year].append(data)

            # Process each year group
            for year, papers in papers_by_year.items():
                texts = [paper[2] + paper[3] for paper in papers]
                vectors = get_embedding(texts, tokenizer, model)
                vectors_with_id = [[paper[0]] + list(vectors[i]) for i, paper in enumerate(papers)]
                save_vectors(vectors_with_id, year, STORAGE, PATH_OUTPUT)


# Get the last processed paper index
last_processed_index = get_last_processed_index(PATH_OUTPUT)
logging.info(f"Resuming from paper {last_processed_index + 1}.")

# --------------------------------------------------------
# Process the papers
# --------------------------------------------------------
process_papers(last_processed_index)

end_total = time.time()
total_duration = end_total - start_total
logging.info("=" * 50)
logging.info(f"Total process finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_total))}")
logging.info(f"Total duration: {total_duration:.2f} seconds ({total_duration / 60:.1f} minutes)")
logging.info("=" * 50)
