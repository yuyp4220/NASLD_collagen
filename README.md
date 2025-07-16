# Collagen Analysis Pipeline

This repository contains the code for quantifying collagen features from histology image tiles.

## Structure

- `core/`: contains all function definitions
- `main.py`: batch process multiple slides and export CSV
- `.gitignore`: excludes large or sensitive data
- `README.md`: this file

## Data Structure

Expected folder layout:

```
data/
├── F0/
│   ├── slide1/
│   │   ├── tile1
│   │   ├── tile2
│   │   └── ...
│   ├── slide2/
│   │   ├── tile1
│   │   ├── tile2
│   │   └── ...
├── F1/
│   ├── slide3/
│   │   ├── tile1
│   │   └── ...
...
```

Where:
- `F0`, `F1`, etc. are fibrosis stages
- Each `slide` folder contains multiple tiles (image or JSONs)

## How to Use

1. Prepare tile folders under `data/` as above
2. Run `main.py` to process and export features
3. (Optional) Run `visualize.py` to inspect individual tiles

## Notes

- All patient data and outputs are kept locally and not uploaded.
- Only code and pipeline logic are included in this repo.
