
# Seventh Step: New Phrases

import csv
from tqdm import tqdm
import collections
import sys
import os
import logging
import time
import pandas as pd


# --------------------------------------------------------
# Configure file paths
# --------------------------------------------------------

target_path = "/data/home/Zhuqian_He/originality" 
os.chdir(target_path)
sys.path.append(target_path)

HISTORICAL_PHRASES_PATH = "Arts_results/papers_phrases.csv"  
Y2024_PHRASES_PATH = "Data/Output/papers_phrases.csv"  
HISTORICAL_NEW_PHRASES_PATH = "Arts_results/new_phrases.csv"  
OUTPUT_PATH = "Data/Output/new_phrases.csv" 

# --------------------------------------------------------
# Configure logging
# --------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('Log/7_New_phrase.log'),
        logging.StreamHandler(sys.stdout)  
    ]
)

start_total = time.time()  
logging.info("=" * 50)
logging.info(f"Total process started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_total))}")
logging.info("=" * 50)


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

chunk_size = 100000

# --------------------------
# 1. Load historical phrases
# --------------------------
logging.info("Loading all phrases up to 2023...")
historical_all_phrases = set()  

with open(HISTORICAL_PHRASES_PATH, 'r', encoding='utf-8') as f:
    total_historical_papers = sum(1 for _ in f) - 1  

with open(HISTORICAL_PHRASES_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')
    next(reader)  
    
    for line in tqdm(reader, total=total_historical_papers, desc="Processing historical phrases"):
        title_phrases = line[1].split()
        abstract_phrases = line[2].split()
        paper_phrases = set(title_phrases + abstract_phrases)  
        historical_all_phrases.update(paper_phrases)  

# --------------------------
# 2. Load new_phrases data up to 2023
# --------------------------
logging.info("\nLoading new_phrases.csv up to 2023...")
historical_new_phrases = {}
row_count = 0

with open(HISTORICAL_NEW_PHRASES_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')
    next(reader)  
    
    for line in reader:
        phrases, first_paper_id, reuse = line
        historical_new_phrases[phrases] = (first_paper_id, int(reuse))
        row_count += 1

logging.info(f"Total rows in new_phrases.csv: {row_count}")

# --------------------------
# 3. Process 2024 data
# --------------------------
reuse_in_2024 = collections.Counter()
new_phrases_2024 = {}  
new_phrases_reuse_2024 = collections.Counter()

logging.info("\nProcessing 2024 data...")
y2024_papers = []
with open(Y2024_PHRASES_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')
    next(reader)  
    for line in reader:
        paper_id = line[0]
        pub_date = line[3]  
        title_phrases = [p for p in line[1].split(', ') if p.strip()]
        abstract_phrases = [p for p in line[2].split(', ') if p.strip()]
        paper_phrases = set(title_phrases + abstract_phrases)
        y2024_papers.append((pub_date, paper_id, paper_phrases))  

y2024_papers_sorted = sorted(y2024_papers, key=lambda x: x[0])
total_2024_papers = len(y2024_papers_sorted)

for pub_date, paper_id, paper_phrases in tqdm(y2024_papers_sorted, total=total_2024_papers, desc="Processing 2024 data (sorted)"):
    for phrase in paper_phrases:  
        if phrase in historical_all_phrases:
            reuse_in_2024[phrase] += 1  
        else:
            if phrase not in new_phrases_2024:
                new_phrases_2024[phrase] = (paper_id, pub_date)
            else:
                new_phrases_reuse_2024[phrase] += 1   

# --------------------------
# 4. Update new_phrases data
# --------------------------
logging.info("\nUpdating new_phrases.csv...")
updated_new_phrases = {}  

for phrase, (first_id, old_reuse) in historical_new_phrases.items():
    updated_reuse = old_reuse + reuse_in_2024.get(phrase, 0)
    updated_new_phrases[phrase] = (first_id, updated_reuse)

for phrase, (first_id, _) in new_phrases_2024.items():
    updated_new_phrases[phrase] = (first_id, new_phrases_reuse_2024.get(phrase, 0)) 

# --------------------------
# 5. Export results
# --------------------------
total_output_rows = 0

with open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Phrase', 'PaperID', 'Reuse'])  
    total_output_rows += 1
    
    for phrase in sorted(updated_new_phrases.keys()):
        first_id, reuse = updated_new_phrases[phrase]
        writer.writerow([phrase, first_id, reuse])
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
