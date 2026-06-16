import pandas as pd
from pathlib import Path

for f in [1, 4, 7]:
    csv_path = f"/d/DoAn/SourceCode/data/raw/Evo200_Mixed{f}.csv"
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*60}")
    print(f"File: Evo200_Mixed{f}.csv (bias issue)")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows (header):")
    print(df.head())
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nBasic stats:\n{df.describe()}")
    
    # Check for gaps in data
    print(f"\nUnique 'Thoi Gian' values (first 50):")
    print(df['Thoi Gian'].value_counts().head(50))
