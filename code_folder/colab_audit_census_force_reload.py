# ============================================================================
# CENSUS AUDIT - Force Reload Version
# ============================================================================
# This version force-reloads audit_data to break Python's module cache
# Run this after updating audit_data.py with %%writefile
# 
# PREREQUISITES:
# - Llama-3 8B model and tokenizer must be loaded in memory (from previous cell)
# - audit_data.py must exist with audit_queries list
# ============================================================================

import os
import sys
import csv
import torch
import importlib
from pathlib import Path
from datetime import datetime

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
# DATA FILE CONFIGURATION
# ============================================================================
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
# RESUME CAPABILITY - Check for existing results
# ============================================================================
results_file = "/content/drive/MyDrive/census_audit_results.csv"
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
# AUDIT LOOP - Process queries
# ============================================================================
print("\n" + "="*70)
print("STARTING AUDIT LOOP")
print("="*70)

processed_count = 0
skipped_count = 0

for idx, query_data in enumerate(audit_queries, 1):
    # Create unique key for this query
    query_key = (
        str(query_data.get('ISO3', '')),
        str(query_data.get('Country', '')),
        str(query_data.get('Metric', ''))
    )
    
    # Skip if already completed
    if query_key in completed_queries:
        skipped_count += 1
        if idx % 100 == 0:
            print(f"[Query {idx}/{total_queries}] Skipped (already completed)")
        continue
    
    # Progress monitoring - Show details every 5th query
    show_details = (idx % 5 == 0) or (idx == 1)
    if show_details:
        print(f"\n[Query {idx}/{total_queries}] Processing...")
        print(f"  Country: {query_data.get('Country', 'Unknown')}")
        print(f"  Metric: {query_data.get('Metric', 'Unknown')[:60]}...")
    else:
        print(f"[Query {idx}/{total_queries}] Processing...", end='\r')
    
    # Extract query text
    query_text = query_data.get('Query', '')
    if not query_text:
        print("  ⚠ No query text found, skipping")
        continue
    
    # Llama-3 Instruction Formatting
    formatted_prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{query_text}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    try:
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate
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
        
        # Decode response (remove input prompt)
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
            ai_response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            # Fallback: remove the input prompt
            ai_response = full_response.replace(formatted_prompt, "").strip()
        
        # Incremental Saving - Append immediately after EACH response
        # This ensures zero data loss even if the script is interrupted
        timestamp = datetime.now().isoformat()
        with open(results_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
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
        
        processed_count += 1
        
        # Progress monitoring - Show sample every 5th query
        if show_details or (processed_count % 5 == 0):
            print(f"\n  ✓ Saved to CSV (Total processed: {processed_count:,})")
            print(f"  AI Response: {ai_response[:200]}...")
            print(f"  Progress: {processed_count:,}/{total_queries:,} ({100*processed_count/total_queries:.1f}%)")
        
        # Mark as completed
        completed_queries.add(query_key)
        
    except Exception as e:
        print(f"  ✗ Error processing query: {e}")
        # Save error response
        timestamp = datetime.now().isoformat()
        with open(results_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
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
print("""
from google.colab import files
files.download('/content/drive/MyDrive/census_audit_results.csv')
""")
print("="*70)
print("\n✅ Audit complete! Results saved with zero data loss.")
print("="*70)
