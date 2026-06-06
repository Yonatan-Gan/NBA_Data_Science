import pandas as pd
import requests
from bs4 import BeautifulSoup
import io
import time
import os

output_dir = "nba_raw_data"
os.makedirs(output_dir, exist_ok=True)

years = range(2021, 2027)
target_tables = ['per_game', 'advanced']

print("Starting League-Wide Playoff Download...")
print("---------------------------------------------------------")

for year in years:
    for table_id in target_tables:
        # Hitting the master playoff URLs instead of team URLs
        url = f"https://www.basketball-reference.com/playoffs/NBA_{year}_{table_id}.html"
        file_name = f"{output_dir}/LEAGUE_PLAYOFFS_{year}_{table_id}.csv"
        
        if os.path.exists(file_name):
            print(f"Already exists: {file_name} -> Skipping.")
            continue
            
        print(f"Fetching League-Wide Playoffs: Year={year} | Table={table_id}")
        
        try:
            response = requests.get(url, timeout=10)
        except Exception as e:
            print(f"Network error: {e}")
            continue
            
        if response.status_code == 429:
             print("CRITICAL: Hit rate limit. Waiting 60s...")
             time.sleep(60)
             continue
             
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')
            
            # The master pages append '_stats' to the HTML IDs
            table = soup.find('table', {'id': f"{table_id}_stats"})
            
            if not table:
                table = soup.find('table', {'id': table_id}) # Fallback
            
            if table:
                df = pd.read_html(io.StringIO(str(table)))[0]
                
                # Clean up the dataframe
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(0)
                    
                first_col = df.columns[0]
                df = df[df[first_col] != first_col] # Remove repeating headers
                df = df[df.isnull().sum(axis=1) != df.shape[1]] # Drop empty rows
                
                df['Season'] = year
                df.to_csv(file_name, index=False)
                print(f"  -> Saved: {file_name}")
            else:
                print(f"  -> Table not found in DOM for {year}.")
        
        time.sleep(4.5)