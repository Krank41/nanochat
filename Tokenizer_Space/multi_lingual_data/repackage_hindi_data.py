"""
Repackage Hindi dataset into standardized parquet shards matching the English dataset format.

This script takes the large Hindi parquet files and:
- Extracts only the 'text' column (removing uuid and metadata)
- Creates ~90MB compressed shards (matching English shard size)
- Uses the same parquet settings as the English dataset
- Maintains row group size of 1024 for consistency

Input: Multi_linguial_DataSet_Pre-Paration/hindi-data/*.parquet
Output: Multi_linguial_DataSet_Pre-Paration/hindi-data-processed/shard_*.parquet
"""

import os
import time
import pyarrow.parquet as pq
import pyarrow as pa

def process_hindi_data():
    # Configuration matching the English dataset
    chars_per_shard = 250_000_000  # ~250M characters per shard (uncompressed)
    row_group_size = 1024  # Same as English dataset

    # Input and output directories
    input_dir = "Multi_linguial_DataSet_Pre-Paration/hindi-data"
    output_dir = "Multi_linguial_DataSet_Pre-Paration/hindi-data-processed"
    os.makedirs(output_dir, exist_ok=True)

    # Get all Hindi parquet files
    hindi_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    hindi_files.sort()  # Process in order

    print(f"Found {len(hindi_files)} Hindi parquet files to process")
    print(f"Files: {hindi_files}")

    # Initialize shard variables
    shard_docs = []
    shard_index = 0
    shard_characters = 0
    total_docs_processed = 0
    total_time_spent = 0

    # Process each Hindi file
    for file_idx, hindi_file in enumerate(hindi_files):
        file_path = os.path.join(input_dir, hindi_file)
        print(f"\nProcessing file {file_idx + 1}/{len(hindi_files)}: {hindi_file}")

        # Read the parquet file
        table = pq.read_table(file_path, columns=['text'])

        # Convert to list of texts
        texts = table.column('text').to_pylist()
        print(f"  - Loaded {len(texts)} documents from {hindi_file}")

        t0 = time.time()

        # Process each document
        for doc_idx, text in enumerate(texts):
            if text is None or not isinstance(text, str):
                continue  # Skip invalid entries

            shard_docs.append(text)
            shard_characters += len(text)

            # Check if we should write a shard
            collected_enough_chars = shard_characters >= chars_per_shard
            docs_multiple_of_row_group = len(shard_docs) % row_group_size == 0

            if collected_enough_chars and docs_multiple_of_row_group:
                # Write shard
                shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
                shard_table = pa.Table.from_pydict({"text": shard_docs})

                pq.write_table(
                    shard_table,
                    shard_path,
                    row_group_size=row_group_size,
                    use_dictionary=False,
                    compression="zstd",
                    compression_level=3,
                    write_statistics=False,
                )

                # Update statistics
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                total_docs_processed += len(shard_docs)
                total_time_spent += dt

                # Get file size in MB
                file_size_mb = os.path.getsize(shard_path) / (1024 * 1024)

                print(f"  Wrote {shard_path}")
                print(f"    - Documents: {len(shard_docs)}")
                print(f"    - Characters: {shard_characters:,}")
                print(f"    - Compressed size: {file_size_mb:.1f} MB")
                print(f"    - Time: {dt:.2f}s")

                # Reset for next shard
                shard_docs = []
                shard_characters = 0
                shard_index += 1

            # Progress indicator every 10000 docs
            if (doc_idx + 1) % 10000 == 0:
                print(f"  - Processed {doc_idx + 1}/{len(texts)} documents from current file")

    # Write any remaining documents as final shard (if they meet minimum requirements)
    if shard_docs and len(shard_docs) >= row_group_size:
        # Pad to multiple of row_group_size if needed
        while len(shard_docs) % row_group_size != 0:
            shard_docs.append("")  # Add empty docs to align

        shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
        shard_table = pa.Table.from_pydict({"text": shard_docs})

        pq.write_table(
            shard_table,
            shard_path,
            row_group_size=row_group_size,
            use_dictionary=False,
            compression="zstd",
            compression_level=3,
            write_statistics=False,
        )

        file_size_mb = os.path.getsize(shard_path) / (1024 * 1024)
        print(f"\nWrote final shard: {shard_path}")
        print(f"  - Documents: {len(shard_docs)}")
        print(f"  - Characters: {shard_characters:,}")
        print(f"  - Compressed size: {file_size_mb:.1f} MB")

        total_docs_processed += len(shard_docs)

    # Final statistics
    print("\n" + "="*50)
    print("Processing Complete!")
    print(f"Total documents processed: {total_docs_processed:,}")
    print(f"Total shards created: {shard_index + (1 if shard_docs else 0)}")
    print(f"Total time: {total_time_spent:.2f}s")
    print(f"Output directory: {output_dir}")

    # Verify all shards
    print("\nVerifying created shards:")
    created_shards = sorted([f for f in os.listdir(output_dir) if f.startswith('shard_')])
    total_size_mb = 0
    for shard in created_shards[:5]:  # Show first 5
        path = os.path.join(output_dir, shard)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        total_size_mb += size_mb
        print(f"  {shard}: {size_mb:.1f} MB")
    if len(created_shards) > 5:
        print(f"  ... and {len(created_shards) - 5} more shards")
        for shard in created_shards[5:]:
            path = os.path.join(output_dir, shard)
            total_size_mb += os.path.getsize(path) / (1024 * 1024)

    print(f"\nTotal size of all shards: {total_size_mb:.1f} MB")

if __name__ == "__main__":
    process_hindi_data()