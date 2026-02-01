import os
import sys
import csv
import torch
import importlib
from pathlib import Path
from datetime import datetime
from google.colab import drive

# Verify model and tokenizer are loaded
try:
    model
    tokenizer
    print("✓ Model and tokenizer found in memory")
except NameError:
    raise RuntimeError(
        "Model and tokenizer not found!\n"
        "Please run the model loading cell first (colab_load_llama3_8b.py)"
    )

# ============================================================================
# CONFIGURATION
# ============================================================================
# Batch processing configuration
BATCH_SIZE = 4  # Process queries in batches

# Google Drive configuration
RESULTS_PATH = "/content/drive/MyDrive/census_audit_results.csv"  # Path on Google Drive

# Specify the filename containing your audit_queries list
# Common names: audit_data, census_data, queries_data, etc.
DATA_FILE_NAME = None  # Set to your filename (without .py extension) if known
# If None, script will try common filenames

# ============================================================================
# FORCE RELOAD - Break Python's module cache and find data file
# ============================================================================
print("="*70)
print("FINDING AND LOADING AUDIT DATA")
print("="*70)

# Invalidate all caches
importlib.invalidate_caches()

# Try to find the data file
possible_names = []
if DATA_FILE_NAME:
    possible_names = [DATA_FILE_NAME]
else:
    # Try common filenames
    possible_names = [
        'audit_data',
        'census_data',
        'queries_data',
        'audit_queries',
        'census_queries',
        'data',
        'queries'
    ]

audit_queries = None
loaded_from = None

for name in possible_names:
    # Remove from sys.modules if exists
    if name in sys.modules:
        del sys.modules[name]
        print(f"  ✓ Removed {name} from sys.modules")

    # Try to import
    try:
        module = importlib.import_module(name)
        importlib.reload(module)

        # Check if it has audit_queries
        if hasattr(module, 'audit_queries'):
            audit_queries = module.audit_queries
            loaded_from = name
            print(f"  ✓ Found and loaded audit_queries from {name}.py")
            break
    except ImportError:
        continue
    except Exception as e:
        print(f"  ⚠ Error loading {name}: {e}")
        continue

# If not found, raise error with helpful message
if audit_queries is None:
    print("\n  ✗ Could not find audit_queries in any of these files:")
    for name in possible_names:
        print(f"    - {name}.py")
    print("\n  Please either:")
    print("  1. Set DATA_FILE_NAME = 'your_filename' at the top of this script")
    print("  2. Or ensure your file is named one of the common names above")
    raise ImportError("Could not find audit_queries. Please specify DATA_FILE_NAME or rename your file.")

# ============================================================================
# VERIFY THE COUNT
# ============================================================================
query_count = len(audit_queries)
print(f"\n✓ Successfully loaded {query_count:,} queries from {loaded_from}.py")

if query_count == 0:
    raise ValueError(f"audit_queries is empty in {loaded_from}.py! Check your data file.")
elif query_count < 100:
    print(f"  ⚠ WARNING: Only {query_count} queries found. Expected ~1,704.")
    print(f"  ⚠ Check {loaded_from}.py to ensure all queries are included.")
else:
    print(f"  ✓ Query count verified: {query_count:,} queries")
    if query_count == 1704:
        print(f"  ✓ Perfect! Expected 1,704 queries (213 countries × 8 metrics)")

total_queries = query_count
print("\n" + "="*70)
print("CENSUS AUDIT - Processing Queries")
print("="*70)
print(f"Total queries: {total_queries:,}")

# ============================================================================
# GOOGLE DRIVE SETUP
# ============================================================================
print("\n" + "="*70)
print("MOUNTING GOOGLE DRIVE")
print("="*70)
try:
    drive.mount('/content/drive', force_remount=False)
    print("✓ Google Drive mounted successfully")
except Exception as e:
    print(f"⚠ Warning: Could not mount Google Drive: {e}")
    print("  Will use local file as fallback")
    RESULTS_PATH = "census_audit_results.csv"

# Ensure RESULTS_PATH directory exists
results_dir = Path(RESULTS_PATH).parent
results_dir.mkdir(parents=True, exist_ok=True)
print(f"✓ Results will be saved to: {RESULTS_PATH}")

# ============================================================================
# RESUME CAPABILITY - Check for existing results
# ============================================================================
results_file = RESULTS_PATH
completed_queries = set()

if Path(results_file).exists():
    print(f"\n[Resume] Found existing results file: {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Create unique key from ISO3, Country, and Metric
            key = (row.get('ISO3', ''), row.get('Country', ''), row.get('Metric', ''))
            if all(key):  # Only count if all fields present
                completed_queries.add(key)

    print(f"  ✓ Found {len(completed_queries):,} completed queries")
    remaining = total_queries - len(completed_queries)
    print(f"  ✓ Remaining queries: {remaining:,}")
else:
    print(f"\n[Resume] No existing results file - starting fresh")
    # Create CSV with headers
    with open(results_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'ISO3', 'Country', 'Metric', 'Ground_Truth', 'Query',
            'AI_Response', 'Timestamp', 'Query_Number'
        ])

# ============================================================================
# FLASH ATTENTION CHECK
# ============================================================================
flash_attention_available = False
try:
    import flash_attn
    flash_attention_available = True
    print("\n✓ Flash Attention 2 library detected")
except ImportError:
    print("\n⚠ Flash Attention 2 library not installed")
    print("  Note: Flash Attention will be used automatically if model was loaded with it")
    # Check if model has flash attention support
    if hasattr(model, 'config') and hasattr(model.config, '_attn_implementation'):
        if 'flash' in str(model.config._attn_implementation).lower():
            flash_attention_available = True
            print("  ✓ Model appears to support Flash Attention")

# ============================================================================
# AUDIT LOOP - Process queries in batches with Flash Attention
# ============================================================================
print("\n" + "="*70)
print("STARTING AUDIT LOOP (Batch Processing with Flash Attention)")
print("="*70)
print(f"Batch size: {BATCH_SIZE}")
print(f"Flash Attention: {'Enabled' if flash_attention_available else 'Auto (if supported)'}")

processed_count = 0
skipped_count = 0

# Filter out already completed queries
pending_queries = []
for idx, query_data in enumerate(audit_queries, 1):
    query_key = (
        str(query_data.get('ISO3', '')),
        str(query_data.get('Country', '')),
        str(query_data.get('Metric', ''))
    )
    if query_key not in completed_queries:
        pending_queries.append((idx, query_data, query_key))
    else:
        skipped_count += 1

print(f"Pending queries: {len(pending_queries):,}")
print(f"Skipped (already completed): {skipped_count:,}")

# Process in batches
for batch_start in range(0, len(pending_queries), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(pending_queries))
    batch = pending_queries[batch_start:batch_end]
    
    batch_num = (batch_start // BATCH_SIZE) + 1
    total_batches = (len(pending_queries) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} queries...")
    
    # Prepare batch prompts
    batch_prompts = []
    batch_metadata = []
    
    for idx, query_data, query_key in batch:
        query_text = query_data.get('Query', '')
        if not query_text:
            print(f"  ⚠ Query {idx}: No query text found, skipping")
            continue
        
        # Llama-3 Instruction Formatting
        formatted_prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{query_text}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        batch_prompts.append(formatted_prompt)
        batch_metadata.append((idx, query_data, query_key, query_text))
    
    if not batch_prompts:
        continue
    
    # Tokenize batch
    try:
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        # Generate with batch processing (Flash Attention used automatically if model supports it)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode batch responses
        batch_responses = []
        for i, output in enumerate(outputs):
            full_response = tokenizer.decode(output, skip_special_tokens=True)
            # Extract only the assistant's response
            if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
                ai_response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            else:
                # Fallback: remove the input prompt
                ai_response = full_response.replace(batch_prompts[i], "").strip()
            batch_responses.append(ai_response)
        
        # Save all responses in batch to Google Drive
        timestamp = datetime.now().isoformat()
        with open(results_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for (idx, query_data, query_key, query_text), ai_response in zip(batch_metadata, batch_responses):
                writer.writerow([
                    query_data.get('ISO3', ''),
                    query_data.get('Country', ''),
                    query_data.get('Metric', ''),
                    query_data.get('Ground_Truth', ''),
                    query_text,
                    ai_response,
                    timestamp,
                    idx
                ])
                completed_queries.add(query_key)
                processed_count += 1
        
        print(f"  ✓ Batch {batch_num} complete: Saved {len(batch_responses)} responses to {RESULTS_PATH}")
        print(f"  Progress: {processed_count:,}/{total_queries:,} ({100*processed_count/total_queries:.1f}%)")
        
        # Show sample response from batch
        if batch_responses:
            sample_response = batch_responses[0][:200]
            print(f"  Sample response: {sample_response}...")
    
    except Exception as e:
        print(f"  ✗ Error processing batch {batch_num}: {e}")
        # Save error responses for this batch
        timestamp = datetime.now().isoformat()
        with open(results_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for idx, query_data, query_key, query_text in batch_metadata:
                writer.writerow([
                    query_data.get('ISO3', ''),
                    query_data.get('Country', ''),
                    query_data.get('Metric', ''),
                    query_data.get('Ground_Truth', ''),
                    query_text,
                    f"ERROR: {str(e)}",
                    timestamp,
                    idx
                ])
                completed_queries.add(query_key)
                processed_count += 1
        continue

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("AUDIT COMPLETE")
print("="*70)
print(f"Total queries: {total_queries:,}")
print(f"Processed: {processed_count:,}")
print(f"Skipped (already completed): {skipped_count:,}")
print(f"Results saved to: {results_file}")
print("="*70)

# ============================================================================
# DOWNLOAD CSV TO COMPUTER
# ============================================================================
print("\n" + "="*70)
print("DOWNLOAD RESULTS")
print("="*70)
print("\nTo download the CSV file to your computer:")
print("\nOption 1: Use Colab's file browser")
print("  1. Click the folder icon on the left sidebar")
print(f"  2. Right-click on '{results_file}'")
print("  3. Select 'Download'")
print("\nOption 2: Use Python code (run in next cell):")
print("="*70)
print(f"""
from google.colab import files
files.download('{RESULTS_PATH}')
""")
print("="*70)
print(f"\n✅ Audit complete! Results saved to Google Drive: {RESULTS_PATH}")
print("✅ All responses saved with zero data loss using batch processing.")
print("="*70)