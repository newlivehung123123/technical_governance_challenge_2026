# ============================================================================
# CENSUS AUDIT - Process 1,816 Queries with Resume Capability
# ============================================================================
# Runs full audit with incremental saving and progress monitoring
# ============================================================================

import os
import csv
import torch
import importlib
from pathlib import Path
from datetime import datetime

# ============================================================================
# SAFETY CHECK - Invalidate caches to see new file immediately
# ============================================================================
importlib.invalidate_caches()

# ============================================================================
# DATA SETUP - Import audit queries from separate file
# ============================================================================
# The audit_queries list is loaded from audit_data.py
# Run the %%writefile audit_data.py cell FIRST, then run this cell
# ============================================================================

from audit_data import audit_queries

total_queries = len(audit_queries)
print("="*70)
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
# AUDIT LOOP - Process queries in batches for A100 optimization
# ============================================================================
print("\n" + "="*70)
print("STARTING AUDIT LOOP (Batch Size: 32)")
print("="*70)
print("✓ Flash Attention 2 enabled (from model loader)")
print("✓ Batch processing enabled for maximum A100 throughput")
print("="*70)

BATCH_SIZE = 32
processed_count = 0
skipped_count = 0

# Filter out completed queries and prepare batch list
pending_queries = []
for idx, query_data in enumerate(audit_queries, 1):
    query_key = (
        str(query_data.get('ISO3', '')),
        str(query_data.get('Country', '')),
        str(query_data.get('Metric', ''))
    )
    
    if query_key in completed_queries:
        skipped_count += 1
        continue
    
    # Add index and query data
    pending_queries.append((idx, query_data, query_key))

print(f"\nPending queries to process: {len(pending_queries):,}")
print(f"Already completed: {skipped_count:,}")

# Process in batches
for batch_start in range(0, len(pending_queries), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(pending_queries))
    batch = pending_queries[batch_start:batch_end]
    batch_num = (batch_start // BATCH_SIZE) + 1
    total_batches = (len(pending_queries) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} queries...")
    
    # Prepare batch data
    batch_indices = []
    batch_prompts = []
    batch_query_data = []
    batch_keys = []
    
    for idx, query_data, query_key in batch:
        query_text = query_data.get('Query', '')
        if not query_text:
            continue
        
        # Llama-3 Instruction Formatting
        formatted_prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{query_text}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        batch_indices.append(idx)
        batch_prompts.append(formatted_prompt)
        batch_query_data.append(query_data)
        batch_keys.append(query_key)
    
    if not batch_prompts:
        continue
    
    try:
        # Tokenize batch with padding and truncation
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Adjust if needed
        ).to(model.device)
        
        # Generate for entire batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,  # Reduced for short answers
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode all responses in batch
        timestamp = datetime.now().isoformat()
        batch_results = []
        
        for i, (idx, query_data, query_key) in enumerate(zip(batch_indices, batch_query_data, batch_keys)):
            try:
                # Decode this specific response (account for padding)
                full_response = tokenizer.decode(outputs[i], skip_special_tokens=True)
                
                # Extract only the assistant's response
                if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
                    ai_response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
                else:
                    # Fallback: remove the input prompt
                    ai_response = full_response.replace(batch_prompts[i], "").strip()
                
                batch_results.append({
                    'idx': idx,
                    'query_data': query_data,
                    'query_key': query_key,
                    'response': ai_response,
                    'error': None
                })
                
            except Exception as e:
                batch_results.append({
                    'idx': idx,
                    'query_data': query_data,
                    'query_key': query_key,
                    'response': f"ERROR: {str(e)}",
                    'error': str(e)
                })
        
        # Save batch results immediately
        with open(results_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for result in batch_results:
                query_data = result['query_data']
                writer.writerow([
                    query_data.get('ISO3', ''),
                    query_data.get('Country', ''),
                    query_data.get('Metric', ''),
                    query_data.get('Ground_Truth', ''),
                    query_data.get('Query', ''),
                    result['response'],
                    timestamp,
                    result['idx']
                ])
                completed_queries.add(result['query_key'])
                processed_count += 1
        
        print(f"  ✓ Batch saved ({processed_count:,} total processed)")
        if batch_results:
            sample = batch_results[0]['response']
            print(f"  Sample response: {sample[:100]}...")
        
    except Exception as e:
        print(f"  ✗ Batch error: {e}")
        # Save individual error entries for this batch
        timestamp = datetime.now().isoformat()
        with open(results_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for idx, query_data, query_key in zip(batch_indices, batch_query_data, batch_keys):
                writer.writerow([
                    query_data.get('ISO3', ''),
                    query_data.get('Country', ''),
                    query_data.get('Metric', ''),
                    query_data.get('Ground_Truth', ''),
                    query_data.get('Query', ''),
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
