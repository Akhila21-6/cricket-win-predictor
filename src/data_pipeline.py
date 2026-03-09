import pandas as pd
import os
import glob

def process_match_data(raw_path, processed_path):
    # Search for all ball-by-ball files
    match_files = glob.glob(os.path.join(raw_path, "[0-9]*.csv"))
    all_processed_matches = []

    print(f"Found {len(match_files)} matches. Processing...")

    for file in match_files:
        if file.endswith('_info.csv'):
            continue  # Skip info files here; we handle them inside the loop
            
        try:
            # 1. Load the ball-by-ball data
            df = pd.read_csv(file)
            
            # Check column names (some files use 'ball' as '1.1' style instead of separate 'over'/'ball')
            if 'over' not in df.columns and 'ball' in df.columns:
                # If 'ball' contains values like 0.1, 0.2, etc.
                df['over_num'] = df['ball'].astype(int)
                df['ball_num'] = ((df['ball'] - df['over_num']) * 10).round().astype(int)
            else:
                df['over_num'] = df['over']
                df['ball_num'] = df['ball']

            # 2. Calculate Features
            df['current_score'] = df.groupby('innings')['runs_off_bat'].cumsum() + \
                                  df.groupby('innings')['extras'].cumsum()
            
            df['wickets_fallen'] = df.groupby('innings')['wicket_type'].transform(lambda x: x.notnull().cumsum())
            df['wickets_remaining'] = 10 - df['wickets_fallen']
            
            df['balls_bowled'] = (df['over_num'] * 6) + df['ball_num']
            df['balls_remaining'] = 120 - df['balls_bowled']

            # 3. Get the Winner from the _info.csv file
            info_file = file.replace(".csv", "_info.csv")
            # Use on_bad_lines='skip' to handle the rows with 4 fields
            info_df = pd.read_csv(info_file, header=None, names=['type', 'key', 'value'], on_bad_lines='skip')
            
            winner_row = info_df[info_df['key'] == 'winner']
            if not winner_row.empty:
                winner = winner_row['value'].values[0]
                batting_team = df['batting_team'].iloc[0]
                df['target'] = 1 if winner == batting_team else 0
                all_processed_matches.append(df)
        
        except Exception as e:
            print(f"Skipping {os.path.basename(file)}: {e}")

    if all_processed_matches:
        final_df = pd.concat(all_processed_matches)
        output_file = os.path.join(processed_path, "cleaned_match_data.csv")
        final_df.to_csv(output_file, index=False)
        print(f"Success! Saved to: {output_file}")
    else:
        print("No data processed. Please check if files in data/raw are empty.")

if __name__ == "__main__":
    RAW_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    process_match_data(RAW_DIR, PROCESSED_DIR)