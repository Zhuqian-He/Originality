
# First Step: Screen Paper between 2019 and 2024
import pandas as pd
import sys
import os
import logging
import time

# --------------------------------------------------------
# Configure file paths
# --------------------------------------------------------
target_path = "/data/home/Zhuqian_He/originality"
os.chdir(target_path)
sys.path.append(target_path)

Raw_data = 'Data/Raw/orig_works.csv.gz'
output_screen_paper = 'Data/Raw/orig_calcu_2024_works.csv'

# --------------------------------------------------------
# Configure logging
# --------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('Log/1_Screen_Paper.log'),
        logging.StreamHandler(sys.stdout)  
    ]
)

chunksize = 100000
filtered_chunks = []  
chunk_idx = 0  

start_total = time.time()  
logging.info("=" * 50)
logging.info(f"Total process started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_total))}")
logging.info(f"Chunk size: {chunksize}, Input file: {Raw_data}")  
logging.info("=" * 50)

# --------------------------------------------------------
# Read files in chunks, filter papers between 2019 and 2024
# --------------------------------------------------------
for chunk in pd.read_csv(Raw_data, chunksize=chunksize, compression='gzip'):
    chunk_idx += 1  
    chunk['publication_year'] = pd.to_numeric(chunk['publication_year'], errors='coerce')
    mask = ((chunk['publication_year'] <= 2024) & (chunk['publication_year'] >= 2019))
    filtered_chunk = chunk.loc[mask]

    filtered_chunks.append(filtered_chunk)
    logging.info(f"Processed chunk {chunk_idx} | Rows retained: {len(filtered_chunk)}") 

if filtered_chunks:
    df = pd.concat(filtered_chunks, ignore_index=True)
else:
    df = pd.read_csv(Raw_data, nrows=0, compression='gzip')

df.reset_index(drop=True, inplace=True) 
logging.info(f"All chunks merged | Total rows: {len(df)}")

# --------------------------------------------------------
# Export results
# --------------------------------------------------------
df.to_csv(output_screen_paper, index=False, header=True, sep=",", encoding="utf-8-sig", na_rep="NULL")
logging.info(f"Output saved to: {output_screen_paper}")

end_total = time.time()
total_duration = end_total - start_total
logging.info("=" * 50)
logging.info(f"Total process finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_total))}")
logging.info(f"Total duration: {total_duration:.2f} seconds ({total_duration / 60:.1f} minutes)")
logging.info("=" * 50)


