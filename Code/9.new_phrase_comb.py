
# Ninth Step: New Phrases Combination

import csv
from tqdm import tqdm
import collections
import sys
import os
import logging
import time
import itertools as it
import pandas as pd
import re
import multiprocessing
from typing import Set, Tuple

# --------------------------------------------------------
# Configure file paths
# --------------------------------------------------------
target_path = "/data/home/Zhuqian_He/originality" 
os.chdir(target_path)
sys.path.append(target_path)

HISTORICAL_PHRASES_PATH = "Arts_results/papers_phrases.csv"  
HISTORICAL_NEW_COMBS_PATH = "Arts_results/new_phrase_combs.csv"  
Y2024_PHRASES_PATH = "Data/Output/papers_phrases.csv"  
OUTPUT_PATH = "Data/Output/new_phrase_combs.csv"  

# --------------------------------------------------------
# Configure logging
# --------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('Log/9_New_phrase_comb.log'),
        logging.StreamHandler(sys.stdout)  
    ]
)

# --------------------------------------------------------
# Global configurations for multiprocessing
# --------------------------------------------------------
phrase_splitter = re.compile(r'\s+')
CHUNK_SIZE = 10000  
PROCESS_NUM = 8  

# --------------------------------------------------------
# Multiprocessing core function
# --------------------------------------------------------
def process_chunk(chunk: pd.DataFrame) -> Set[Tuple[str, str]]:
    chunk_combs = set()
    for idx, row in chunk.iterrows():
        title = row.iloc[0]
        abstract = row.iloc[1]
        
        all_phrases = []
        if pd.notna(title):
            all_phrases.extend([p.strip() for p in phrase_splitter.split(title)])
        if pd.notna(abstract):
            all_phrases.extend([p.strip() for p in phrase_splitter.split(abstract)])
        paper_phrases = frozenset([p for p in all_phrases if p])
        for comb in it.combinations(paper_phrases, 2):
            chunk_combs.add(tuple(sorted(comb)))
    return chunk_combs

# --------------------------------------------------------
# Main process
# --------------------------------------------------------
if __name__ == "__main__":
    start_total = time.time()  
    logging.info("=" * 50)
    logging.info(f"Total process started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_total))}")
    logging.info("=" * 50)

    ## Increase the max size of a line reading, otherwise an error is raised
    maxInt = sys.maxsize
    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)

    # --------------------------
    # 1. Load historical data
    # --------------------------
    logging.info("Loading historical phrase combinations up to 2023...")
    historical_all_combs = set()  

    try:
        with open(HISTORICAL_PHRASES_PATH, 'r', encoding='utf-8') as f:
            total_historical_papers = sum(1 for _ in f) - 1  
    except Exception as e:
        logging.error(f"Failed to get total rows of historical phrases: {e}")
        total_historical_papers = None

    df_chunks = pd.read_csv(
        HISTORICAL_PHRASES_PATH,
        encoding='utf-8',
        usecols=[1, 2],  
        skiprows=1, 
        dtype=str,  
        chunksize=CHUNK_SIZE,  
        on_bad_lines='skip'  
    )

    total_chunks = (total_historical_papers // CHUNK_SIZE) + 1 if total_historical_papers else None
    with multiprocessing.Pool(processes=PROCESS_NUM, maxtasksperchild=1) as pool:
        for chunk_combs in tqdm(
            pool.imap(process_chunk, df_chunks),
            total=total_chunks,
            desc="Processing historical phrase combinations"
        ):
            historical_all_combs.update(chunk_combs)

    processed_count = total_historical_papers if total_historical_papers else len(historical_all_combs)  

    '''
    with open(HISTORICAL_PHRASES_PATH, 'r', encoding='utf-8') as f:
        total_historical_papers = sum(1 for _ in f) - 1  

    with open(HISTORICAL_PHRASES_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        next(reader)  
    
        for line in tqdm(reader, total=total_historical_papers, desc="Processing historical phrase combinations"):
            title_phrases = [w.strip() for w in line[1].split() if w.strip()]
            abstract_phrases = [w.strip() for w in line[2].split() if w.strip()]
            paper_phrases = set(title_phrases + abstract_phrases)
        
            combs = list(it.combinations(paper_phrases, 2))
            sorted_combs = set([tuple(sorted(comb)) for comb in combs])
        
            historical_all_combs.update(sorted_combs)
    '''

    # --------------------------
    # 2. Load new_phrases_comb data up to 2023
    # --------------------------
    logging.info("\nLoading historical new_phrase_combs.csv up to 2023...")
    historical_new_combs = {}  
    row_count = 0

    with open(HISTORICAL_NEW_COMBS_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        next(reader)  
    
        for line in reader:
            phrase1, phrase2, first_paper_id, reuse = line
            comb = tuple(sorted([phrase1, phrase2]))
            historical_new_combs[comb] = (first_paper_id, int(reuse))
            row_count += 1

    logging.info(f"Total rows in historical new_phrase_combs.csv: {row_count}")

    # --------------------------
    # 3. Process 2024 data
    # --------------------------
    reuse_in_2024 = collections.Counter()
    new_combs_2024 = {}  
    new_combs_reuse_2024 = collections.Counter()

    logging.info("\nProcessing 2024 paper phrase combinations...")
    y2024_papers = []
    with open(Y2024_PHRASES_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        next(reader)  
    
        for line in reader:
            paper_id = line[0]
            pub_date = line[3]  
            title_phrases = [w.strip() for w in line[1].split() if w.strip()]
            abstract_phrases = [w.strip() for w in line[2].split() if w.strip()]
            paper_phrases = set(title_phrases + abstract_phrases)
        
            y2024_papers.append((pub_date, paper_id, paper_phrases))

    y2024_papers_sorted = sorted(y2024_papers, key=lambda x: x[0])
    total_2024_papers = len(y2024_papers_sorted)

    for pub_date, paper_id, paper_phrases in tqdm(
        y2024_papers_sorted, 
        total=total_2024_papers, 
        desc="Processing 2024 combinations (sorted)"
    ):
        combs = list(it.combinations(paper_phrases, 2))
        sorted_combs = set([tuple(sorted(comb)) for comb in combs])
    
        for comb in sorted_combs:
            if comb in historical_all_combs:
                reuse_in_2024[comb] += 1
            else:
                if comb not in new_combs_2024:
                    new_combs_2024[comb] = (paper_id, pub_date)
                else:
                    new_combs_reuse_2024[comb] += 1

    # --------------------------
    # 4. Update new_wphrase_comb data
    # --------------------------
    logging.info("\nUpdating new_phrase_combs.csv...")
    updated_new_combs = {}

    for comb, (first_id, old_reuse) in historical_new_combs.items():
        updated_reuse = old_reuse + reuse_in_2024.get(comb, 0)
        updated_new_combs[comb] = (first_id, updated_reuse)

    for comb, (first_id, _) in new_combs_2024.items():
        updated_new_combs[comb] = (first_id, new_combs_reuse_2024.get(comb, 0))

    # --------------------------
    # 5. Export results
    # --------------------------
    total_output_rows = 0

    with open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Phrase1', 'Phrase2', 'PaperID', 'Reuse'])  
        total_output_rows += 1
    
        for comb in sorted(updated_new_combs.keys()):
            phrase1, phrase2 = comb
            first_id, reuse = updated_new_combs[comb]
            writer.writerow([phrase1, phrase2, first_id, reuse])
            total_output_rows += 1

    logging.info(f"Total rows in output file: {total_output_rows}") 
    logging.info(f"Completed! Updated results saved to {OUTPUT_PATH}")

    end_total = time.time()
    total_duration = end_total - start_total
    logging.info("=" * 50)
    logging.info(f"Total process finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_total))}")
    logging.info(f"Total duration: {total_duration:.2f} seconds ({total_duration / 60:.1f} minutes)")
    logging.info(f"Output file row count: {total_output_rows}") 
    logging.info("=" * 50)
