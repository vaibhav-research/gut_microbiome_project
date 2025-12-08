import os
from pathlib import Path
from tqdm import tqdm
import torch
import pandas as pd
import pyarrow.parquet as pq
from data_loading import (
    generate_dna_embeddings_h5, 
    generate_microbiome_embeddings_h5, 
    sanity_check_dna_and_microbiome_embeddings,
    load_labels
)

# ==================== Configuration Constants ====================
# Define these here to avoid dependency on global config.yaml
# You might want to adjust these paths if your environment differs

SRS_TO_OTU_PARQUET = Path("data_preprocessing/mapref_data/samples-otus-97.parquet")
OTU_TO_DNA_PARQUET = Path("data_preprocessing/mapref_data/otus_97_to_dna.parquet")
MICROBIOME_CHECKPOINT = Path("data/checkpoint_epoch_0_final_epoch3_conf00.pt")
EMBEDDING_MODEL = "neuralbioinfo/prokbert-mini-long"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # or mps optionally

def build_paths(base_dir: Path, csv_path: Path):
    """
    Construct output paths based on the dataset structure.
    Assumes csv_path is like .../dataset_name/csv_file.csv
    Output structure:
      base_dir/sequences/dataset_name/csv_stem/...
      base_dir/dna_embeddings/dataset_name/csv_stem/...
      base_dir/microbiome_embeddings/dataset_name/csv_stem/...
    """
    # Use the CSV filename (without extension) as the sub-folder name
    csv_stem = csv_path.stem 
    dataset_name = csv_path.parent.name
    
    sequences_dir = base_dir / "sequences" / dataset_name / csv_stem
    dna_embeddings_dir = base_dir / "dna_embeddings" / dataset_name / csv_stem
    microbiome_embeddings_dir = base_dir / "microbiome_embeddings" / dataset_name / csv_stem
    
    return sequences_dir, dna_embeddings_dir, microbiome_embeddings_dir

def optimized_generate_dna_csvs(sids: list, srs_to_otu_parquet: Path, otu_to_dna_parquet: Path, sequences_dir: Path):
    """
    Optimized version of generate_dna_csvs_for_samples that loads data in bulk
    to avoid N_samples * N_otus file reads.
    """
    
    # 1. Check which SIDs actually need generation to avoid redundant work
    missing_sids = [sid for sid in sids if not (sequences_dir / f"{sid}.csv").exists()]
    
    if not missing_sids:
        print(f"  All {len(sids)} DNA CSVs already exist, skipping generation")
        return

    print(f"  Generating {len(missing_sids)} missing DNA CSV files...")
    sequences_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load OTU -> DNA mapping (small enough for memory ~80MB on disk)
    print("  Loading OTU to DNA mapping...")
    try:
        otu_dna_df = pd.read_parquet(otu_to_dna_parquet)
        # Convert to dictionary for O(1) access: otu_id -> dna_sequence
        # Assuming columns are 'otu_97_id' and 'dna_sequence' based on data_loading.py checks
        otu_to_dna_map = dict(zip(otu_dna_df['otu_97_id'], otu_dna_df['dna_sequence']))
        del otu_dna_df # Free memory
    except Exception as e:
        print(f"  Error loading OTU parquet: {e}")
        return

    # 3. Process SIDs in batches to respect memory while avoiding single-row reads
    # 'srs_to_otu_parquet' can be large (2.4GB). 
    # We filter by 'srs_id' in missing_sids using pyarrow filters which is efficient.
    
    batch_size = 500 
    
    for i in tqdm(range(0, len(missing_sids), batch_size), desc="  Processing sample batches"):
        batch_sids = missing_sids[i : i + batch_size]
        
        try:
            # Read parquet filtering for this batch of SIDs
            # filters=[('srs_id', 'in', batch_sids)] is efficient in PyArrow
            table = pq.read_table(srs_to_otu_parquet, filters=[('srs_id', 'in', batch_sids)])
            batch_df = table.to_pandas()
            
            # Group by srs_id
            grouped = batch_df.groupby('srs_id')
            
            found_sids = set()
            
            for srs_id, group in grouped:
                found_sids.add(srs_id)
                
                # Get OTUs
                otu_ids = group['otu_id'].tolist()
                
                # Map to DNA
                data = []
                for otu in otu_ids:
                    # Look up DNA sequence from pre-loaded map
                    dna = otu_to_dna_map.get(otu)
                    if dna:
                        data.append({'otu_id': otu, 'dna_sequence': dna})
                
                # Create DataFrame
                if data:
                    df_out = pd.DataFrame(data)
                else:
                     df_out = pd.DataFrame(columns=['otu_id', 'dna_sequence'])
                
                # Save
                df_out.to_csv(sequences_dir / f"{srs_id}.csv", index=False)
            
            # Handle SIDs that were not found in parquet (0 OTUs)
            for sid in batch_sids:
                if sid not in found_sids:
                    # Create empty CSV
                    pd.DataFrame(columns=['otu_id', 'dna_sequence']).to_csv(sequences_dir / f"{sid}.csv", index=False)
                    
        except Exception as e:
            print(f"  Error processing batch starting with {batch_sids[0]}: {e}")
            import traceback
            traceback.print_exc()

def generate_embeddings(csv_path: Path, base_dir: Path):
    print(f"\nProcessing dataset: {csv_path}")
    sequences_dir, dna_embeddings_dir, microbiome_embeddings_dir = build_paths(base_dir, csv_path)
    
    # 0. Load SIDs
    try:
        labels_dict = load_labels(csv_path)
        sids = list(labels_dict.keys())
        print(f"  Found {len(sids)} samples")
    except Exception as e:
        print(f"  Error loading labels from {csv_path}: {e}")
        return

    # 1. Extract/Generate DNA CSVs
    # We call the generation function directly for the list of SIDs
    if not sequences_dir.exists() or not any(sequences_dir.iterdir()):
        print(f"  Generating DNA CSVs in {sequences_dir}...")
        optimized_generate_dna_csvs(
            sids,
            SRS_TO_OTU_PARQUET,
            OTU_TO_DNA_PARQUET,
            sequences_dir
        )
    else:
        print(f"  DNA CSVs already exist at {sequences_dir}, skipping generation")

    # 2. Generate DNA embeddings (ProkBERT) -> H5
    dna_embeddings_h5_path = dna_embeddings_dir / 'dna_embeddings.h5'
    if not dna_embeddings_h5_path.exists():
        print(f"  Generating DNA embeddings H5 in {dna_embeddings_dir}...")
        generate_dna_embeddings_h5(
            sequences_dir, 
            dna_embeddings_dir,  # Note: logic inside appends / 'dna_embeddings.h5' if passing dir? 
                                 # wait, data_loading.generate_dna_embeddings_h5 takes output_h5_path as second arg
            # Let's check data_loading.py signature: 
            # def generate_dna_embeddings_h5(dna_csv_dir: Path, output_h5_path: Path, ...)
            # And inside it does: output_h5_path_file = output_h5_path / 'dna_embeddings.h5'
            # So pass the directory!
            EMBEDDING_MODEL,
            batch_size=BATCH_SIZE,
            device=DEVICE
        )
    else:
        print(f"  DNA embeddings already exist at {dna_embeddings_h5_path}, skipping generation")

    # 3. Generate Microbiome Embeddings (Transformer) -> H5
    microbiome_embeddings_h5_path = microbiome_embeddings_dir / 'microbiome_embeddings.h5'
    if not microbiome_embeddings_h5_path.exists():
        print(f"  Generating Microbiome embeddings H5 in {microbiome_embeddings_dir}...")
        # Check signature: 
        # def generate_microbiome_embeddings_h5(prokbert_h5_path: Path, output_h5_path: Path, checkpoint_path: Path, ...)
        # It also does output_h5_path_file = output_h5_path / 'microbiome_embeddings.h5'
        
        generate_microbiome_embeddings_h5(
            dna_embeddings_h5_path,
            microbiome_embeddings_dir, # Pass directory
            MICROBIOME_CHECKPOINT,
            device=DEVICE
        )
    else:
        print(f"  Microbiome embeddings already exist at {microbiome_embeddings_h5_path}, skipping generation")

    # 4. Sanity Check
    try:
        # Check signature: def sanity_check_dna_and_microbiome_embeddings(dna_embeddings_dir: Path, microbiome_embeddings_dir: Path):
        # This function expects directories containing the .h5 files
        if sanity_check_dna_and_microbiome_embeddings(dna_embeddings_dir, microbiome_embeddings_dir):
             print("  ✓ Data sanity check passed")
    except Exception as e:
        print(f"  x Sanity check failed: {e}")

    return

if __name__ == "__main__":
    # Define input and output directories
  
    BASE_OUTPUT_DIR = Path("YOUR_OUTPUT_DIR") # where output will be saved
    DATASET_DIR = Path("YOUR_DATASET_DIR") # directory containing the CSV files

    if not DATASET_DIR.exists():
        print(f"Dataset directory {DATASET_DIR} does not exist. Please check path.")
    else:
        # loop over all CSV files in the directory
        print(f"Scanning for CSVs in {DATASET_DIR}...")
        files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".csv")]
        
        if not files:
            print("No CSV files found.")
        
        for file in files:
            csv_path = DATASET_DIR / file
            generate_embeddings(csv_path, BASE_OUTPUT_DIR)