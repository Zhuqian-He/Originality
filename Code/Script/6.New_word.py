
# Sixth Step: New Words

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

HISTORICAL_WORDS_PATH = "Arts_results/papers_words.csv"  
Y2024_WORDS_PATH = "Data/Output/papers_words.csv"  
HISTORICAL_NEW_WORDS_PATH = "Arts_results/new_words.csv"  
OUTPUT_PATH = "Data/Output/new_words.csv" 

# --------------------------------------------------------
# Configure logging
# --------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('Log/6_New_word.log'),
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
# 1. Load historical words
# --------------------------
logging.info("Loading all words up to 2023...")
historical_all_words = set()  

with open(HISTORICAL_WORDS_PATH, 'r', encoding='utf-8') as f:
    total_historical_papers = sum(1 for _ in f) - 1  

with open(HISTORICAL_WORDS_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')
    next(reader)  
    
    for line in tqdm(reader, total=total_historical_papers, desc="Processing historical words"):
        title_words = line[1].split()
        abstract_words = line[2].split()
        paper_words = set(title_words + abstract_words)  
        historical_all_words.update(paper_words)  

# --------------------------
# 2. Load new_words data up to 2023
# --------------------------
logging.info("\nLoading new_words.csv up to 2023...")
historical_new_words = {}
row_count = 0

with open(HISTORICAL_NEW_WORDS_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')
    next(reader)  
    
    for line in reader:
        word, first_paper_id, reuse = line
        historical_new_words[word] = (first_paper_id, int(reuse))
        row_count += 1

logging.info(f"Total rows in new_words.csv: {row_count}")

# --------------------------
# 3. Process 2024 data
# --------------------------
reuse_in_2024 = collections.Counter()
new_words_2024 = {}
new_words_reuse_2024 = collections.Counter()

logging.info("\nProcessing 2024 data...")
y2024_papers = []
with open(Y2024_WORDS_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')
    next(reader)  
    for line in reader:
        paper_id = line[0]
        pub_date = line[3]  
        title_words = line[1].split()
        abstract_words = line[2].split()
        paper_words = set(title_words + abstract_words)
        y2024_papers.append((pub_date, paper_id, paper_words))  

y2024_papers_sorted = sorted(y2024_papers, key=lambda x: x[0])
total_2024_papers = len(y2024_papers_sorted)

for pub_date, paper_id, paper_words in tqdm(y2024_papers_sorted, total=total_2024_papers, desc="Processing 2024 data (sorted)"):
    for word in paper_words:
        if word in historical_all_words:
            reuse_in_2024[word] += 1  
        else:
            if word not in new_words_2024:
                new_words_2024[word] = (paper_id, pub_date)
            else:
                new_words_reuse_2024[word] += 1  

# --------------------------
# 4. Update new_words data
# --------------------------
logging.info("\nUpdating new_words.csv...")
updated_new_words = {}

for word, (first_id, old_reuse) in historical_new_words.items():
    updated_reuse = old_reuse + reuse_in_2024.get(word, 0)
    updated_new_words[word] = (first_id, updated_reuse)

for word, (first_id, _) in new_words_2024.items():
    updated_new_words[word] = (first_id, new_words_reuse_2024.get(word, 0)) 

# --------------------------
# 5. Export results
# --------------------------
total_output_rows = 0

with open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Word', 'PaperID', 'Reuse'])  
    total_output_rows += 1
    
    for word in sorted(updated_new_words.keys()):
        first_id, reuse = updated_new_words[word]
        writer.writerow([word, first_id, reuse])
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
