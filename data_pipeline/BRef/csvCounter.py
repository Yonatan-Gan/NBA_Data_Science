import os
import pandas as pd

def count_scraped_data(folder_path="nba_raw_data"):
    """
    Scans the specified folder, counts the total number of CSV files, 
    the total number of rows (records), and the total file size.
    """
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist in the current directory.")
        return

    total_rows = 0
    total_files = 0
    total_size_bytes = 0

    print(f"Scanning folder: {folder_path}...\n")

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            
            # Add to total file size
            file_size = os.path.getsize(filepath)
            total_size_bytes += file_size
            total_files += 1

            # Count rows using Pandas
            try:
                df = pd.read_csv(filepath)
                total_rows += len(df)
            except Exception as e:
                print(f"Could not read {filename}: {e}")

    # Convert bytes to megabytes
    total_size_mb = total_size_bytes / (1024 * 1024)

    # Print the final report
    print("=========================================================")
    print("Basketball-Reference Data Summary")
    print("=========================================================")
    print(f"Total CSV Files:      {total_files}")
    print(f"Total Records (Rows): {total_rows:,}")
    print(f"Total Size:           {total_size_mb:.2f} MB")
    print("=========================================================")

if __name__ == "__main__":
    count_scraped_data()