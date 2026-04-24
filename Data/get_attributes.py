import os
import pandas as pd

def generate_csv_summary(output_file="csv_contents_summary.txt"):
    # List all CSV files in the current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("DATASET COLUMN SUMMARY REPORT\n")
        f.write("=" * 40 + "\n\n")
        
        for file_name in csv_files:
            f.write(f"FILE: {file_name}\n")
            f.write("-" * 40 + "\n")
            
            try:
                # Read only the first two rows (header + 1 row of data) to save memory
                df = pd.read_csv(file_name, nrows=1)
                
                if df.empty:
                    f.write("Status: File is empty or has no data rows.\n")
                else:
                    f.write(f"Total Columns found: {len(df.columns)}\n")
                    f.write("Entries / Columns List:\n")
                    
                    for col in df.columns:
                        # Get a sample value to show what kind of data is in the column
                        sample_val = df[col].iloc[0] if len(df) > 0 else "N/A"
                        f.write(f"  - {col}  (Example: {sample_val})\n")
            
            except Exception as e:
                f.write(f"Error reading file: {str(e)}\n")
            
            f.write("\n" + "*" * 50 + "\n\n")
            print(f"Processed: {file_name}")

    print(f"\nSuccess! Summary saved to: {output_file}")

if __name__ == "__main__":
    generate_csv_summary()