import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
import io
import time
import os

def scrape_bref_team_data(team_abbr, year, table_id):
    """
    Scrapes a specific data table from a team's Basketball-Reference page.
    """
    # The 'games' (schedule) table lives on a separate sub-page
    if table_id == 'games':
        url = f"https://www.basketball-reference.com/teams/{team_abbr}/{year}_games.html"
    else:
        url = f"https://www.basketball-reference.com/teams/{team_abbr}/{year}.html"
    
    try:
        response = requests.get(url, timeout=10)
    except Exception as e:
        print(f"Network error fetching {team_abbr} for {year}: {e}")
        return None
    
    if response.status_code == 404:
        print(f"Skipping: {team_abbr} did not exist in {year}.")
        return None
    elif response.status_code == 429:
        print("CRITICAL: Hit rate limit (429). Waiting 60 seconds...")
        time.sleep(60) 
        return None
    elif response.status_code != 200:
        print(f"Error {response.status_code} for {team_abbr} {year}.")
        return None

    # Basketball-Reference may use alternate table IDs.
    possible_ids = [table_id]
    if table_id == 'per_game': 
        possible_ids.extend(['per_game_stats', 'per_game-team'])
    elif table_id == 'playoffs_per_game': 
        possible_ids.extend(['playoffs_per_game_stats', 'playoffs_per_game-team'])
    elif table_id == 'playoffs_advanced': 
        possible_ids.extend(['playoffs_advanced_stats', 'playoffs_advanced-team'])
    elif table_id == 'salaries': 
        possible_ids.extend(['salaries2']) # Alternate ID used for the salaries table
    elif table_id == 'pbp': 
        possible_ids.extend(['pbp_stats', 'play_by_play'])
    elif table_id == 'advanced':
        possible_ids.extend(['advanced_stats'])

    # Load normal DOM 
    soup = BeautifulSoup(response.text, 'lxml')
    table = None
    
    # Search the visible HTML for any recognized table ID.
    for pid in possible_ids:
        table = soup.find('table', {'id': pid})
        if table:
            break
            
    # If not found, search for tables embedded in HTML comments.
    if not table:
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            # Parse only comments that may contain a table.
            if '<table' in comment:  
                comment_soup = BeautifulSoup(comment, 'lxml')
                for pid in possible_ids:
                    table = comment_soup.find('table', {'id': pid})
                    if table:
                        break
            if table:
                break

    if table is None:
        # A missing playoff table may indicate that the team did not participate.
        print(f"  -> Table '{table_id}' not found for {team_abbr} ({year}). (Skipped)")
        return None
        
    try:
        # Wrap the HTML in io.StringIO to remove the Pandas FutureWarning
        df = pd.read_html(io.StringIO(str(table)))[0]
        
        # Clean up multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
            
        # Drop completely empty rows
        df = df[df.isnull().sum(axis=1) != df.shape[1]] 
        return df
    except Exception as e:
        print(f"Error parsing table {table_id} for {team_abbr} ({year}): {e}")
        return None

# =====================================================================
# Main Automation Configuration
# =====================================================================
if __name__ == "__main__":
    
    output_dir = "nba_raw_data"
    os.makedirs(output_dir, exist_ok=True)
    
    nba_teams = [
        'ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
        'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
        'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
    ]
    
    start_year = 2021
    end_year = 2026
    years = list(range(start_year, end_year + 1))
    
    target_tables = [
        'roster',             # Roster construction, age, exp 
        'team_misc',          # ORTG, DRTG, Pace 
        'per_game',           # Regular season raw stats 
        'advanced',           # Regular season advanced stats 
        'playoffs_per_game',  # Playoff raw stats 
        'playoffs_advanced',  # Playoff advanced stats 
        'games',              # Schedule, rest days, back-to-backs
        'salaries',           # Financial roster hierarchy
        'pbp'                 # On/Off court net impact
    ]
    
    print(f"Starting bulk download for years {start_year}-{end_year}...")
    print(f"Total planned requests: {len(nba_teams) * len(years) * len(target_tables)}")
    print("---------------------------------------------------------")

    for year in years:
        for team in nba_teams:
            for table_id in target_tables:
                
                file_name = f"{output_dir}/{team}_{year}_{table_id}.csv"
                
                if os.path.exists(file_name):
                    print(f"Already exists: {file_name} -> Skipping.")
                    continue
                
                print(f"Processing: Team={team} | Year={year} | Table={table_id}")
                
                df = scrape_bref_team_data(team, year, table_id)
                
                if df is not None and not df.empty:
                    # Clean up repeating header rows common in B-Ref
                    # Usually "Rk" or the first column name repeats every 20 rows
                    first_col = df.columns[0]
                    df = df[df[first_col] != first_col]
                    
                    df['Team'] = team
                    df['Season'] = year
                    df.to_csv(file_name, index=False)
                    print(f"  -> Saved: {file_name}")
                
                # Rate limit safety (wait 4.5 seconds to ensure < 20 requests/min)
                time.sleep(4.5)
                
    print("\n=========================================================")
    print("Bulk scraping sequence complete! Data is ready for analysis.")
    print("=========================================================")