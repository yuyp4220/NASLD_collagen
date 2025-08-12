import os
import pandas as pd
from pathlib import Path
from functions.core import aggregate_area_and_pct_features_with_portal  

parent_path = Path('data folder path')
base_paths = [str(p) for p in parent_path.glob("**/*") if p.is_dir() and p.name.endswith("jsons")]

all_dfs = []

for path in base_paths:
    path_obj = Path(path)
    patient_id = path_obj.parts[-2]  # e.g. 2139504_A546_B2006...
    stage = path_obj.parts[-3]       # e.g. F0, F1, etc.
    
    df = aggregate_area_and_pct_features_with_portal(path)
    df["ID"] = patient_id
    df["label"] = stage
    all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)
pivot_df = combined_df.pivot(index=["label", "ID"], columns="Feature", values="Aggregated Value").reset_index()
pivot_df.to_csv("output_structured_features.csv", index=False)