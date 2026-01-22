# merge_to_parquet.py
import os
import sys
from pathlib import Path
import humanize

# ---------- CONFIG ----------
DATA_DIR = Path("./data/raw")
OUTPUT_DIR = Path("./data/processed")  
GLOB_PATTERN = "*.csv"
PARTITION_BY_LABEL = True 



def list_files(d):
    files = sorted([p for p in d.glob(GLOB_PATTERN) if p.is_file()])
    if not files:
        print(f"No CSV files found in {d}")
        sys.exit(1)
    print(f"Found {len(files)} CSV files in {d}:")
    for f in files:
        try:
            size = humanize.naturalsize(f.stat().st_size)
        except Exception:
            size = "unknown"
        print(f" - {f.name}  ({size})")
    return files

def main():
    files = list_files(DATA_DIR)

    try:
        import dask.dataframe as dd
        import pyarrow  # ensure pyarrow available
        print("\nUsing Dask to read CSVs...")
        file_paths = [str(p) for p in files]
        df = dd.read_csv(file_paths, assume_missing=True, dtype="object", low_memory=False)
        print("Columns detected:", df.columns.tolist())
        print("Number of rows (computing may be slow): attempting approximate count...")
        try:
            print("  partitions:", df.npartitions)
        except Exception:
            pass

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if PARTITION_BY_LABEL and 'Label' in df.columns:
            print(f"Writing to parquet partitioned by 'Label' in {OUTPUT_DIR} ...")
            df.to_parquet(str(OUTPUT_DIR), engine="pyarrow", write_index=False, partition_on=["Label"])
        else:
            print(f"Writing to parquet (no partition) in {OUTPUT_DIR} ...")
            df.to_parquet(str(OUTPUT_DIR), engine="pyarrow", write_index=False)

        print("Done. Parquet files are in:", OUTPUT_DIR)
        print("You can read them later with pandas.read_parquet or dd.read_parquet.")
        return

    except Exception as e:
        print("Dask approach failed or not available:", e)
        print("Falling back to chunked pandas concatenation...")

    try:
        import pandas as pd
        out_path = OUTPUT_DIR
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        parts = []
        first_cols = None
        chunk_size = 100_000
        for f in files:
            print("Reading", f)
            it = pd.read_csv(f, chunksize=chunk_size, low_memory=False)
            for i, chunk in enumerate(it):
                if first_cols is None:
                    first_cols = chunk.columns.tolist()
                else:
                    if set(chunk.columns) != set(first_cols):
                        cols_union = list(dict.fromkeys(first_cols + list(chunk.columns)))
                        chunk = chunk.reindex(columns=cols_union)
                part_fp = OUTPUT_DIR / f"{f.stem}_part{i}.parquet"
                chunk.to_parquet(part_fp, engine="pyarrow", index=False)
                print("  wrote", part_fp.name)
        print("Pandas chunked write complete. You have multiple parquet parts in", OUTPUT_DIR)
        return
    except Exception as e:
        print("Pandas fallback also failed:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
