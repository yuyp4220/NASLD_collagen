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

#-----------------------single folder processing----------------------------
import os
import json
import warnings
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

input_root = Path('h:\weekly reports\week 36\F0')
image_exts = [".jpg", ".jpeg"]

slide_folders = [f for f in input_root.iterdir() if f.is_dir()]
for slide_folder in tqdm(slide_folders, desc="Slides Progress"):
    try:
        selected_ranges, slide_is_red = extract_global_hue_range_with_spread(str(slide_folder))
        output_folder = slide_folder / "feature_jsons"
        output_folder.mkdir(exist_ok=True)

        image_list = [p for p in slide_folder.iterdir() if p.suffix.lower() in image_exts and not p.name.startswith("._")]


        for img_path in tqdm(image_list, desc=f"Tiles in {slide_folder.name}", leave=False):
            try:
                params = get_params(str(img_path), selected_ranges, slide_is_red)
                result = extract_basic_collagen_features(**params)

                # 👉 添加 selected_ranges 和 slide_is_red 到 result 中
                result["selected_ranges"] = selected_ranges
                result["slide_is_red"] = slide_is_red

                json_filename = output_folder / (img_path.stem + ".json")
                with open(json_filename, "w") as f:
                    json.dump(result, f, indent=2)

            except Exception as e:
                print(f"❌ Error processing {img_path.name} in {slide_folder.name}: {e}")

    except Exception as e:
        print(f"❌ Error processing slide {slide_folder.name}: {e}")
