
# Fifth Step: Text Processing
import sys
import os
import logging
import time
from tqdm import tqdm
import csv
import numpy as np


# --------------------------------------------------------
# Configure file paths
# --------------------------------------------------------

target_path = "/data/home/Zhuqian_He/originality" 
os.chdir(target_path)
sys.path.append(target_path)

sys.path.insert(1, '/data/home/Zhuqian_He/originality')

import Code.Arts.preprocessing

# --------------------------------------------------------
# Configure logging
# --------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('Log/5_Text_Processing.log'),
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
        

logging.info('Get the number of papers to process...')
with open('Data/Raw/filter_paper.csv', 'r', encoding = 'utf-8') as file:
    line_count = sum(1 for line in file)

# Subtract 1 for the header if the CSV has a header
total_papers = line_count - 1

logging.info('Preparing for writing...')
words_file = open('Data/Output/papers_words.csv', mode='w', encoding='utf-8', newline='')
words_writer = csv.writer(words_file)  
words_writer.writerow(['PaperID', 'Words_Title', 'Words_Abstract', 'publication_date'])  # write the first line for the headers

phrases_file = open('Data/Output/papers_phrases.csv', mode='w', encoding='utf-8', newline='')
phrases_writer = csv.writer(phrases_file) 
phrases_writer.writerow(['PaperID', 'Phrases_Title', 'Phrases_Abstract', 'publication_date']) # write the first line for the headers

logging.info('Processing...')

def clean_text(text):
    if isinstance(text, float) and np.isnan(text):
        return ""
    
    text_str = str(text)
    
    text_str = text_str.replace('\n', '').replace('\r', '')  
    text_str = ' '.join(text_str.split())  
    
    text_str = text_str.replace('"', '')    
    text_str = text_str.replace('“', '')    
    text_str = text_str.replace('”', '')    
    text_str = text_str.replace("'", "")   
    text_str = text_str.replace("‘", "")    
    text_str = text_str.replace("’", "")   
    
    return text_str

with open('Data/Raw/filter_paper.csv', mode = 'r', encoding='utf-8') as reader:
    csv_reader = csv.reader(reader, delimiter=',', quotechar='"')
    
    # Skip header
    next(csv_reader)

    for line in tqdm(csv_reader, total = total_papers):
        
        paperID, date, title, abstract = line
        
        title_clean = str(title).lower()
        abstract_clean = str(abstract).lower()

        title_words = Code.Arts.preprocessing.process_text(title_clean, 'words')
        abstract_words = Code.Arts.preprocessing.process_text(abstract_clean, 'words')

        title_phrases = Code.Arts.preprocessing.process_text(title_clean, 'phrases')
        abstract_phrases = Code.Arts.preprocessing.process_text(abstract_clean, 'phrases')
        
        title_words = clean_text(title_words)
        abstract_words = clean_text(abstract_words)
        title_phrases = clean_text(title_phrases)
        abstract_phrases = clean_text(abstract_phrases)
        
        words_writer.writerow([paperID, title_words, abstract_words, date])
        phrases_writer.writerow([paperID, title_phrases, abstract_phrases, date])

words_file.close()
phrases_file.close()

end_total = time.time()
total_duration = end_total - start_total
logging.info("=" * 50)
logging.info(f"Total process finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_total))}")
logging.info(f"Total duration: {total_duration:.2f} seconds ({total_duration / 60:.1f} minutes)")
logging.info("=" * 50)  
