
# Second Step: Filter Paper between 2019 and 2024
import pandas as pd
import json
import re
import os
import sys
import logging
import time
import swifter
from glob import glob  

# --------------------------------------------------------
# Configure file paths
# --------------------------------------------------------
target_path = "/data/home/Zhuqian_He/originality" 
os.chdir(target_path)
sys.path.append(target_path)

input_raw_data = 'Data/Raw/orig_calcu_2024_works.csv' 
output_filter_paper = 'Data/Raw/filter_paper.csv'   
temp_dir = "temp_all_data"  
chunksize = 100000 

os.makedirs(os.path.dirname(output_filter_paper), exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# --------------------------------------------------------
# Configure logging
# --------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('Log/2_Filter_Paper_Detailed.log'),
        logging.StreamHandler(sys.stdout)  
    ]
)

start_total = time.time()  
logging.info("=" * 50)
logging.info(f"Total process started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_total))}")
logging.info(f"Chunksize: {chunksize}, Input file: {input_raw_data}")
logging.info("=" * 50)


# --------------------------------------------------------
# Read and merge all raw data in chunks
# --------------------------------------------------------
step_name = "1. Read and merge all raw data"
start_step = time.time()
logging.info(f"\n{step_name} started")

chunk_iter = pd.read_csv(input_raw_data, chunksize=chunksize)
chunk_idx = 0
temp_files = []
for chunk in chunk_iter:
    chunk_idx += 1
    temp_file = f"{temp_dir}/raw_chunk_{chunk_idx}.csv"
    chunk.to_csv(temp_file, index=False, encoding="utf-8-sig")
    temp_files.append(temp_file)
logging.info(f"Saved {chunk_idx} raw chunks to temporary directory")

temp_files.sort(key=lambda x: int(re.findall(r'raw_chunk_(\d+)\.csv', x)[0]))
df = pd.concat([pd.read_csv(f) for f in temp_files], ignore_index=True)
total_raw = len(df)
logging.info(f"Total raw data rows after merging: {total_raw}")

for f in temp_files:
    os.remove(f)
os.rmdir(temp_dir)
logging.info("Temporary raw data files cleaned up")

end_step = time.time()
logging.info(f"{step_name} completed. Elapsed: {end_step - start_step:.2f}s")


# --------------------------------------------------------
# Clean title and publication_date
# --------------------------------------------------------
step_name = "Step0: Clean title and publication_date"
start_step = time.time()
logging.info(f"\n{step_name} started")

prev_count = len(df)

df["title"] = df["title"].apply(
    lambda x: pd.NA if isinstance(x, str) and x.strip().upper() in ["NULL", ""] else x
)
df = df.dropna(subset=["title"])
current_count = len(df)
dropped_title = prev_count - current_count
logging.info(f"Rows dropped due to invalid title (NaN/Null/Empty): {dropped_title}")
prev_count = current_count

df["publication_date"] = df["publication_date"].apply(
    lambda x: pd.NA if isinstance(x, str) and x.strip().upper() in ["NULL", ""] else x
)
df = df.dropna(subset=["publication_date"])
current_count = len(df)
dropped_year = prev_count - current_count
logging.info(f"Rows dropped due to invalid publication_date (NaN/Null/Non-numeric): {dropped_year}")
logging.info(f"Remaining rows after {step_name}: {current_count}")

end_step = time.time()
logging.info(f"{step_name} completed. Elapsed time: {end_step - start_step:.2f}s")


# --------------------------------------------------------
# Extract PaperID
# --------------------------------------------------------
step_name = "Step1: Extract PaperID"
start_step = time.time()
logging.info(f"\n{step_name} started")

df['PaperID'] = df['id'].str.split('/').str[-1].str

end_step = time.time()
logging.info(f"{step_name} completed. Elapsed time: {end_step - start_step:.2f}s")


# --------------------------------------------------------
# Convert inverted index to abstract
# --------------------------------------------------------
step_name = "Step2: Convert inverted index to abstract"
start_step = time.time()
logging.info(f"\n{step_name} started")

prev_count = len(df)

def str_to_inverted_dict(inverted_str):
    if not isinstance(inverted_str, str) or inverted_str.strip().lower() in ["", "null", "none"]:
        return {}
    try:
        return json.loads(inverted_str)
    except Exception:
        try:
            return json.loads(inverted_str.replace("'", '"'))
        except Exception:
            return {}

def plain_text_from_inverted(inverted_index):
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""
    positions = []
    http_pattern = re.compile(r'https?://', re.IGNORECASE)
    for word, indices in inverted_index.items():
        if http_pattern.search(word):
            continue
        if isinstance(indices, list):
            for index in indices:
                positions.append((int(index), word))
    positions.sort(key=lambda x: x[0])
    return ' '.join([word for _, word in positions])

df["abstract"] = df["abstract_inverted_index"].swifter.progress_bar(True).apply(
    lambda x: plain_text_from_inverted(str_to_inverted_dict(x))
)

# Statistics on parsing results
success_count = df["abstract"].notna().sum()
failed_count = prev_count - success_count
non_empty_count = (df["abstract"] != "").sum()
empty_count = success_count - non_empty_count

logging.info(f"Abstract parsing: Total {prev_count}, successful {success_count}, failed {failed_count}")
logging.info(f"Non-empty abstracts: {non_empty_count}, empty: {empty_count}")

end_step = time.time()
logging.info(f"{step_name} completed. Elapsed time: {end_step - start_step:.2f}s")


# --------------------------------------------------------
# Exclude non-English papers
# --------------------------------------------------------
step_name = "Step3: Exclude non-English papers"
start_step = time.time()
logging.info(f"\n{step_name} started")

prev_count = len(df)

import spacy
import spacy_fastlang
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.add_pipe("language_detector")

def detect_language_row(text):
    if pd.isna(text) or not isinstance(text, str):
        return "unknown"
    doc = nlp(text)
    return doc._.language if doc._.language is not None else "unknown"

# Detect languages
df["languages_title"] = df["title"].swifter.progress_bar(True).apply(detect_language_row)
df["languages_abstract"] = df["abstract"].swifter.progress_bar(True).apply(detect_language_row)

# Filter non-English
df = df[(df['languages_abstract'].isin(["en", "unknown"])) & (df['languages_title'] == "en")]
current_count = len(df)
dropped_non_english = prev_count - current_count
logging.info(f"Rows dropped due to non-English title/abstract: {dropped_non_english}")
logging.info(f"Remaining rows after {step_name}: {current_count}")

end_step = time.time()
logging.info(f"{step_name} completed. Elapsed time: {end_step - start_step:.2f}s")


# --------------------------------------------------------
# Exclude papers with abstract/summary keywords
# --------------------------------------------------------
step_name = "Step4: Exclude papers with abstract/summary keywords"
start_step = time.time()
logging.info(f"\n{step_name} started")

prev_count = len(df)

abstract_keywords = [
    "Abstract", "Resumen", "摘要", "Résumé", "Resumo", 
    "Streszczenie", "アブストラクト", "Abusutorakuto", 
    "Аннотация", "Özet", "Abstrakt", "Samenvatting", 
    "الملخص", "Al-Mukhtasar", "Abstractum"
]

summary_keywords = [
    "Zusammenfassung", "Resumen", "总结", "RSommaire", "Resumo", 
    "Podsumowanie", "サマリー", "Samarii", "Сводка", "Резюме", 
    "Özet", "Összegzés", "Shrnutí", "Riassunto", "Samenvatting",
    "الملخص", "Al-Mukhtasar", "Summarium"
]

def has_keywords(text, keywords):
    if pd.isna(text) or not isinstance(text, str):
        return False  
    text_lower = text.lower()
    return "abstract" in text_lower and any(keyword.lower() in text_lower for keyword in keywords)

# Filter keywords
filter_abstract = df["abstract"].swifter.progress_bar(True).apply(
    lambda x: has_keywords(x, abstract_keywords)
)
filter_summary = df["abstract"].swifter.progress_bar(True).apply(
    lambda x: has_keywords(x, summary_keywords)
)
df = df[~filter_abstract & ~filter_summary]
current_count = len(df)
dropped_keywords = prev_count - current_count
logging.info(f"Rows dropped due to abstract/summary keywords: {dropped_keywords}")
logging.info(f"Remaining rows after {step_name}: {current_count}")

end_step = time.time()
logging.info(f"{step_name} completed. Elapsed time: {end_step - start_step:.2f}s")


# --------------------------------------------------------
# Exclude bibliographic data in abstracts
# --------------------------------------------------------
step_name = "Step5: Exclude bibliographic data in abstracts"
start_step = time.time()
logging.info(f"\n{step_name} started")

prev_count = len(df)

expression1 = r'\b(?:citation|cited|scholar|altmetric|volume|crossref|scopus|citing|reference|online|cite|download|facebook|twitter|share|pdf|issue|mendeley|article|link)\b.*?\b(?:citation|cited|scholar|altmetric|volume|crossref|scopus|citing|reference|online|cite|download|facebook|twitter|share|pdf|issue|mendeley|article|link)\b.*?\b(?:citation|cited|scholar|altmetric|volume|crossref|scopus|citing|reference|online|cite|download|facebook|twitter|share|pdf|issue|mendeleyarticle|link)\b'
expression2 = r'^(?:articlecontributions|log in|keywords|back to table|previous articlenext|return to|advertisement|book review|no accessjournal|get pdf email|our website|views icon|no other journal|research article|article free|essay\|)'
expression3 = r'^.{0,9}journal' 

def replace_abstract_withna(text):
    if not isinstance(text, str) or pd.isna(text):
        return False
    return (re.search(expression1, text, re.IGNORECASE) is not None or  
            re.search(expression2, text, re.IGNORECASE) is not None or  
            re.search(expression3, text, re.IGNORECASE) is not None)

# Mark abstracts to be filtered
df["abstract"] = df["abstract"].where(
    ~df["abstract"].swifter.progress_bar(True).apply(replace_abstract_withna),  
    other=pd.NA)

# Count marked abstracts
abstract_na_count = df["abstract"].isna().sum()
logging.info(f"Abstracts marked as NA due to bibliographic data: {abstract_na_count}")

end_step = time.time()
logging.info(f"{step_name} completed. Elapsed time: {end_step - start_step:.2f}s")


# --------------------------------------------------------
# Remove duplicate titles
# --------------------------------------------------------
step_name = "Step6: Remove duplicate titles (keep most recent)"
start_step = time.time()
logging.info(f"\n{step_name} started")

prev_count = len(df)

df["title_cleaned"] = df["title"].str.replace(r'[^a-zA-Z0-9]', '', regex=True).str.lower()
df_sorted = df.sort_values(by="publication_date", ascending=False)
df = df_sorted.drop_duplicates(subset="title_cleaned", keep="first")
current_count = len(df)
dropped_duplicate_titles = prev_count - current_count
logging.info(f"Rows dropped due to duplicate titles: {dropped_duplicate_titles}")
logging.info(f"Remaining rows after {step_name}: {current_count}")

end_step = time.time()
logging.info(f"{step_name} completed. Elapsed time: {end_step - start_step:.2f}s")


# --------------------------------------------------------
# Remove duplicate abstracts
# --------------------------------------------------------
step_name = "Step7: Remove duplicate abstracts (keep most recent)"
start_step = time.time()
logging.info(f"\n{step_name} started")

prev_count = len(df)

df["is_no_abstract"] = df["abstract"].apply(
    lambda x: True if pd.isna(x) or str(x).strip().upper() in ["", "NULL"] else False
)
df["abstract_cleaned"] = df.apply(
    lambda row: re.sub(r'[^a-zA-Z0-9]', '', str(row["abstract"]).strip()).lower() 
    if not row["is_no_abstract"] else pd.NA, axis=1
)

df_sorted = df.sort_values(by="publication_date", ascending=False)
df_abstract_nan = df_sorted[df_sorted["is_no_abstract"] == True]
df_abstract_notnan = df_sorted[df_sorted["is_no_abstract"] == False].drop_duplicates(
    subset="abstract_cleaned", keep="first")
df_final = pd.concat([df_abstract_nan, df_abstract_notnan], ignore_index=True)
df_final = df_final.reindex(columns=['PaperID', 'publication_date', 'title', 'abstract'])
current_count = len(df_final)
dropped_duplicate_abstracts = prev_count - current_count

logging.info(f"Rows with missing abstracts: {len(df_abstract_nan)}")
logging.info(f"Rows with unique abstracts: {len(df_abstract_notnan)}")
logging.info(f"Rows dropped due to duplicate abstracts: {dropped_duplicate_abstracts}")
logging.info(f"Remaining rows after {step_name}: {current_count}")

end_step = time.time()
logging.info(f"{step_name} completed. Elapsed time: {end_step - start_step:.2f}s")


### Export results
df_final.to_csv(
    output_filter_paper,
    columns=['PaperID', 'publication_date', 'title', 'abstract'],
    index=False,
    header=True,
    sep=",",
    encoding="utf-8-sig",
    na_rep="NULL"
)
logging.info(f"\nFinal output saved to {output_filter_paper} (total rows: {len(df_final)})")


end_total = time.time()
total_duration = end_total - start_total

logging.info("\n" + "="*50)
logging.info("Filtering summary:")
logging.info(f"- Rows dropped due to invalid title: {dropped_title}")
logging.info(f"- Rows dropped due to invalid year: {dropped_year}")
logging.info(f"- Rows dropped due to non-English: {dropped_non_english}")
logging.info(f"- Rows dropped due to keywords: {dropped_keywords}")
logging.info(f"- Rows dropped due to duplicate titles: {dropped_duplicate_titles}")
logging.info(f"- Rows dropped due to duplicate abstracts: {dropped_duplicate_abstracts}")
logging.info(f"- Total retained rows: {len(df_final)}")
logging.info("="*50 + "\n")