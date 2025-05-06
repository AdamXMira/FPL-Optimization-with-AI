import pandas as pd
import os

DATA_DIR = "data"
SEASON_FILES = ['2024-2025.csv', '2023-2024.csv']

def load_and_clean_season_data():

# Set up the dataframe and loading the data for the model
    dfs = []
    for filename in SEASON_FILES:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        df = pd.read_csv(path)
        df['season'] = filename.replace('.csv', '')
        df['full_name'] = df['first_name'] + ' ' + df['second_name']
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # Drop irrelevant/redundant columns
    cols_to_drop = [
        'first_name', 'second_name', 'creativity', 'influence', 'ict_index',
        'threat', 'bps',
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    # Drop players with 0 points and less than 300 minutes played each season for the last 2 seasons
    # This is to avoid players who have not played enough
    # or have not scored any points
    df = df[df['total_points'] > 0]
    df = df[df['minutes'] > 300] 

    # Fill NaNs in numeric columns with 0
    # This is to avoid NaN values affecting the model
    # and to ensure all players are included
    # in the optimization
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(0)

    # Encode categorical variables: GK=0, DEF=1, MID=2, FWD=3 
    # This is to ensure the model can use these variables
    position_map = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
    df['position'] = df['element_type'].map(position_map)
    df.drop(columns=['element_type'], inplace=True)

    return df

# 'Load and clean' function call
df = load_and_clean_season_data()

# Save cleaned data to CSV
df.to_csv("cleaned_player_data.csv", index=False)

# Preview first few rows and size of the preprocessed dataset
print("Cleaned shape:", df.shape)
print(df.head())


