
# Tenth Step: Calculate paper level metrics

import csv
import os
from collections import defaultdict
from tqdm import tqdm
import logging
import time

# --------------------------------------------------------
# Configure file paths
# --------------------------------------------------------
target_path = "/data/home/Zhuqian_He/originality"
os.chdir(target_path)

NEW_WORDS_PATH = "Data/Output/new_words.csv"
NEW_PHRASES_PATH = "Data/Output/new_phrases.csv"
NEW_WORD_COMBS_PATH = "Data/Output/new_word_combs.csv"
NEW_PHRASE_COMBS_PATH = "Data/Output/new_phrase_combs.csv"
COSINE_PATH = "Data/Output/papers_cosine.csv"
OUTPUT_PATH = "Data/Output/papers_textual_metrics.csv"

# --------------------------------------------------------
# Configure logging
# --------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('Log/7_Combine_Stats.log'),
        logging.StreamHandler()
    ]
)

start_total = time.time()  
logging.info("=" * 50)
logging.info(f"Total process started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_total))}")
logging.info("=" * 50)


# --------------------------------------------------------
# 1. Initialize statistics dictionary (aggregated by PaperID)
# --------------------------------------------------------
stats = defaultdict(lambda: {
    'new_word': 0,
    'new_word_reuse': 0,
    'new_phrase': 0,
    'new_phrase_reuse': 0,
    'new_word_comb': 0,
    'new_word_comb_reuse': 0,
    'new_phrase_comb': 0,
    'new_phrase_comb_reuse': 0,
    'semantic_distance': None  
})


# --------------------------------------------------------
# 2. Process new_words.csv: Count new words and total reuse per PaperID
# --------------------------------------------------------
logging.info("Processing new_words.csv...")
with open(NEW_WORDS_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)  
    total = sum(1 for _ in f) - 1  
    f.seek(0)  
    next(reader)  
    
    for row in tqdm(reader, total=total, desc="Processing new words"):
        paper_id = row['PaperID']
        reuse = int(row['Reuse'])
        
        stats[paper_id]['new_word'] += 1
        stats[paper_id]['new_word_reuse'] += reuse


# --------------------------------------------------------
# 3. Process new_phrases.csv: Count new phrases and total reuse per PaperID
# --------------------------------------------------------
logging.info("Processing new_phrases.csv...")
with open(NEW_PHRASES_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)  
    total = sum(1 for _ in f) - 1
    f.seek(0)
    next(reader)
    
    for row in tqdm(reader, total=total, desc="Processing new phrases"):
        paper_id = row['PaperID']
        reuse = int(row['Reuse'])
        
        stats[paper_id]['new_phrase'] += 1
        stats[paper_id]['new_phrase_reuse'] += reuse


# --------------------------------------------------------
# 4. Process new_word_combs.csv: Count new word combinations and total reuse per PaperID
# --------------------------------------------------------
logging.info("Processing new_word_combs.csv...")
with open(NEW_WORD_COMBS_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)  
    total = sum(1 for _ in f) - 1
    f.seek(0)
    next(reader)
    
    for row in tqdm(reader, total=total, desc="Processing new word combinations"):
        paper_id = row['PaperID']
        reuse = int(row['Reuse'])
        
        stats[paper_id]['new_word_comb'] += 1
        stats[paper_id]['new_word_comb_reuse'] += reuse


# --------------------------------------------------------
# 5. Process new_phrase_combs.csv: Count new phrase combinations and total reuse per PaperID
# --------------------------------------------------------
logging.info("Processing new_phrase_combs.csv...")
with open(NEW_PHRASE_COMBS_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)  
    total = sum(1 for _ in f) - 1
    f.seek(0)
    next(reader)
    
    for row in tqdm(reader, total=total, desc="Processing new phrase combinations"):
        paper_id = row['PaperID']
        reuse = int(row['Reuse'])
        
        stats[paper_id]['new_phrase_comb'] += 1
        stats[paper_id]['new_phrase_comb_reuse'] += reuse


# --------------------------------------------------------
# 6. Process papers_cosine.csv: Supplement semantic distance
# --------------------------------------------------------
logging.info("Processing papers_cosine.csv...")
with open(COSINE_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)  
    total = sum(1 for _ in f) - 1
    f.seek(0)
    next(reader)
    
    for row in tqdm(reader, total=total, desc="Supplementing semantic distance"):
        paper_id = row['PaperID']
        semantic_distance = row['semantic_distance']  
        
        stats[paper_id]['semantic_distance'] = semantic_distance


# --------------------------------------------------------
# 7. Export combined results
# --------------------------------------------------------
logging.info("Exporting combined results...")
with open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as f:
    headers = [
        'PaperID',
        'new_word',
        'new_word_reuse',
        'new_phrase',
        'new_phrase_reuse',
        'new_word_comb',
        'new_word_comb_reuse',
        'new_phrase_comb',
        'new_phrase_comb_reuse',
        'semantic_distance'
    ]
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    
    for paper_id in tqdm(sorted(stats.keys(), key=lambda x: x), desc="Writing results"):
        data = stats[paper_id]
        writer.writerow({
            'PaperID': paper_id,
            'new_word': data['new_word'],
            'new_word_reuse': data['new_word_reuse'],
            'new_phrase': data['new_phrase'],
            'new_phrase_reuse': data['new_phrase_reuse'],
            'new_word_comb': data['new_word_comb'],
            'new_word_comb_reuse': data['new_word_comb_reuse'],
            'new_phrase_comb': data['new_phrase_comb'],
            'new_phrase_comb_reuse': data['new_phrase_comb_reuse'],
            'semantic_distance': data['semantic_distance'] or ''
        })


end_total = time.time()
total_duration = end_total - start_total
logging.info("=" * 50)
logging.info(f"Total process finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_total))}")
logging.info(f"Total duration: {total_duration:.2f} seconds ({total_duration / 60:.1f} minutes)")
logging.info("=" * 50)
