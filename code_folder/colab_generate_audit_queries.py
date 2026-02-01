# ============================================================================
# GENERATE AUDIT QUERIES - Single Cell for Google Colab
# ============================================================================
# This cell generates 1,816 audit queries and saves to audit_data.py
# Run this cell ONCE before running the audit script
# ============================================================================

import pandas as pd
from pathlib import Path

# ============================================================================
# IDENTIFY THE FILE - Load countries from GAID dataset
# ============================================================================
print("="*70)
print("LOADING COUNTRIES FROM GAID DATASET")
print("="*70)

# Try to find the dataset file
dataset_paths = [
    Path("GAID_MASTER_V2_COMPILATION_FINAL.csv"),
    Path("GAID_w1_v2_dataset/GAID_MASTER_V2_COMPILATION_FINAL.csv"),
    Path("../GAID_w1_v2_dataset/GAID_MASTER_V2_COMPILATION_FINAL.csv"),
    Path("/content/GAID_MASTER_V2_COMPILATION_FINAL.csv"),  # Colab upload location
]

dataset_path = None
for path in dataset_paths:
    if path.exists():
        dataset_path = path
        print(f"✓ Found dataset: {dataset_path}")
        break

if dataset_path is None:
    raise FileNotFoundError(
        "GAID_MASTER_V2_COMPILATION_FINAL.csv not found!\n"
        "Please upload the dataset to Colab or place it in the working directory."
    )

# Load dataset and extract unique countries for 2025
print(f"\nLoading dataset...")
print("  Attempting to load with error handling...")

# Track skipped rows using a custom callback
rows_skipped = []
can_track_skipped = False  # Flag to indicate if we can track skipped rows

def track_bad_line(bad_line):
    """Callback to track skipped rows"""
    rows_skipped.append(bad_line)
    return None  # Skip the row

# Try loading with resilience parameters
df = None

try:
    # First attempt: Use on_bad_lines callback to track skipped rows (pandas >= 1.3.0)
    try:
        rows_skipped = []  # Reset counter
        can_track_skipped = True
        df = pd.read_csv(
            dataset_path,
            encoding='utf-8',
            on_bad_lines=track_bad_line,
            engine='python'
        )
        print(f"  ✓ Loaded with on_bad_lines callback")
        if rows_skipped:
            print(f"  ⚠ Skipped {len(rows_skipped)} malformed rows")
    except (TypeError, ValueError):
        # Fallback: Use on_bad_lines='skip' without tracking
        try:
            rows_skipped = []  # Reset counter
            can_track_skipped = False
            df = pd.read_csv(
                dataset_path,
                encoding='utf-8',
                on_bad_lines='skip',
                engine='python'
            )
            print("  ✓ Loaded with on_bad_lines='skip' (skipped rows not counted)")
        except TypeError:
            # Fallback for older pandas: use error_bad_lines=False (deprecated but works)
            try:
                rows_skipped = []  # Reset counter
                can_track_skipped = False
                df = pd.read_csv(
                    dataset_path,
                    encoding='utf-8',
                    error_bad_lines=False,
                    warn_bad_lines=True,
                    engine='python'
                )
                print("  ✓ Loaded with error_bad_lines=False (older pandas)")
                print("  ⚠ Skipped rows count not available with this method")
            except TypeError:
                # Last resort: use sep=None to let pandas guess
                rows_skipped = []  # Reset counter
                can_track_skipped = False
                df = pd.read_csv(
                    dataset_path,
                    encoding='utf-8',
                    sep=None,
                    engine='python',
                    on_bad_lines='skip'
                )
                print("  ✓ Loaded with sep=None (auto-detect)")
                print("  ⚠ Skipped rows count not available with this method")
except Exception as e:
    # Try with different encoding
    print(f"  ⚠ UTF-8 failed: {e}")
    print("  Trying latin1 encoding...")
    rows_skipped = []  # Reset counter
    try:
        can_track_skipped = True
        df = pd.read_csv(
            dataset_path,
            encoding='latin1',
            on_bad_lines=track_bad_line,
            engine='python'
        )
        print("  ✓ Loaded with latin1 encoding")
        if rows_skipped:
            print(f"  ⚠ Skipped {len(rows_skipped)} malformed rows")
    except (TypeError, ValueError):
        # Fallback to skip without tracking
        can_track_skipped = False
        df = pd.read_csv(
            dataset_path,
            encoding='latin1',
            on_bad_lines='skip',
            engine='python'
        )
        print("  ✓ Loaded with latin1 encoding (skipped rows not counted)")
    except Exception as e2:
        raise RuntimeError(f"Could not load dataset: {e2}")

# Report loading results
print(f"✓ Loaded {len(df):,} total rows")
if can_track_skipped and rows_skipped:
    print(f"  ⚠ Skipped {len(rows_skipped)} malformed rows during parsing")
    print(f"  ⚠ This may affect your country count if skipped rows contained 2025 data")
elif not can_track_skipped:
    print(f"  ℹ Skipped rows were not counted (using fallback parsing method)")

# ============================================================================
# DATASET AUDIT - Inspect columns, types, and values
# ============================================================================
print("\n" + "="*70)
print("DATASET AUDIT - Column Inspection")
print("="*70)

# Step 1: Column Inspection
print("\n1. COLUMN NAMES:")
all_columns = df.columns.tolist()
print(f"   Total columns: {len(all_columns)}")
print(f"   Columns: {all_columns}")

# Find Year column (case-insensitive)
year_col = None
for col in all_columns:
    if col.lower() in ['year', 'years', 'yr', 'y']:
        year_col = col
        print(f"   ✓ Found Year column: '{year_col}'")
        break

if year_col is None:
    print("   ⚠ WARNING: No 'Year' column found!")
    print("   ⚠ Searching for similar columns...")
    # Try to find any column that might contain years
    for col in all_columns:
        sample_values = df[col].dropna().head(10).astype(str).tolist()
        if any('202' in str(v) for v in sample_values):
            print(f"   ℹ Potential year column: '{col}' (contains '202' in sample)")
            if year_col is None:
                year_col = col
                print(f"   ✓ Using '{year_col}' as Year column")

if year_col is None:
    raise RuntimeError("Could not identify Year column. Please check your dataset.")

# Step 2: Data Type Check
print(f"\n2. DATA TYPE CHECK for '{year_col}':")
print(f"   dtype: {df[year_col].dtype}")
print(f"   Sample values (first 10 non-null):")
sample_values = df[year_col].dropna().head(10)
for idx, val in enumerate(sample_values, 1):
    print(f"      [{idx}] {repr(val)} (type: {type(val).__name__})")

# Step 3: Unique Values
print(f"\n3. UNIQUE VALUES in '{year_col}':")
unique_years = df[year_col].dropna().unique()
print(f"   Total unique values: {len(unique_years)}")
print(f"   Sorted unique values: {sorted(unique_years)[:20]}")  # Show first 20
if len(unique_years) > 20:
    print(f"   ... (showing first 20 of {len(unique_years)})")
    print(f"   Last values: {sorted(unique_years)[-5:]}")

# Check if 2025 exists in any form
print(f"\n4. SEARCHING FOR 2025:")
year_2025_found = False
for year_val in unique_years:
    year_str = str(year_val).strip()
    if '2025' in year_str or year_str == '2025':
        print(f"   ✓ Found 2025 as: {repr(year_val)} (type: {type(year_val).__name__})")
        year_2025_found = True

if not year_2025_found:
    print("   ⚠ WARNING: '2025' not found in unique values!")
    print("   ⚠ Will attempt type-agnostic search...")

# ============================================================================
# SMART FILTER - Type-agnostic search for 2025
# ============================================================================
print("\n" + "="*70)
print("SMART FILTER - Type-agnostic search for 2025")
print("="*70)

# Convert Year column to string and search for 2025
print(f"\nFiltering for year 2025 using type-agnostic method...")

# Method 1: Try exact match as integer
df_2025_int = df[df[year_col].astype(str).str.strip() == '2025'].copy()
print(f"   Method 1 (exact string match): {len(df_2025_int)} rows")

# Method 2: Try contains '2025' (handles cases like "2025.0", "2025-01-01", etc.)
df_2025_contains = df[df[year_col].astype(str).str.contains('2025', na=False)].copy()
print(f"   Method 2 (contains '2025'): {len(df_2025_contains)} rows")

# Method 3: Try numeric conversion
try:
    df_2025_numeric = df[pd.to_numeric(df[year_col], errors='coerce') == 2025].copy()
    print(f"   Method 3 (numeric == 2025): {len(df_2025_numeric)} rows")
except Exception as e:
    print(f"   Method 3 (numeric) failed: {e}")
    df_2025_numeric = pd.DataFrame()

# Choose the method that found the most rows
df_2025 = None
method_used = None

if len(df_2025_contains) > 0:
    df_2025 = df_2025_contains
    method_used = "contains '2025'"
elif len(df_2025_int) > 0:
    df_2025 = df_2025_int
    method_used = "exact string match"
elif len(df_2025_numeric) > 0:
    df_2025 = df_2025_numeric
    method_used = "numeric == 2025"
else:
    df_2025 = pd.DataFrame()
    method_used = "none (0 rows found)"

print(f"\n✓ Using method: {method_used}")
print(f"✓ Found {len(df_2025):,} rows for year 2025")

# Check for potential data quality issues
if len(df_2025) == 0:
    print("\n" + "⚠"*35)
    print("⚠ WARNING: No rows found for year 2025!")
    print("⚠"*35)
    print("   Possible issues:")
    print("   1. The dataset may not contain 2025 data")
    print("   2. The Year column may use a different format (e.g., '2025-01-01')")
    print("   3. The Year column may be named differently")
    print("\n   Please check the unique values shown above and adjust the filter if needed.")
    raise RuntimeError("Cannot proceed without 2025 data. Please verify your dataset.")

# Get unique ISO3 and Country pairs
print("\n" + "="*70)
print("EXTRACTING UNIQUE COUNTRIES")
print("="*70)

# Find ISO3 and Country columns (case-insensitive)
iso3_col = None
country_col = None

for col in all_columns:
    col_lower = col.lower()
    if col_lower in ['iso3', 'iso_3', 'iso', 'country_code', 'code']:
        if iso3_col is None:
            iso3_col = col
            print(f"✓ Found ISO3 column: '{iso3_col}'")
    elif col_lower in ['country', 'country_name', 'nation', 'territory']:
        if country_col is None:
            country_col = col
            print(f"✓ Found Country column: '{country_col}'")

if iso3_col is None:
    raise RuntimeError("Could not identify ISO3 column. Please check your dataset.")
if country_col is None:
    raise RuntimeError("Could not identify Country column. Please check your dataset.")

print(f"\nExtracting unique countries from columns: '{iso3_col}' and '{country_col}'...")
countries_df = df_2025[[iso3_col, country_col]].drop_duplicates().sort_values(iso3_col)

# Filter out invalid ISO3 codes (should be 3 characters)
print(f"   Before filtering: {len(countries_df)} unique country pairs")
valid_mask = (
    countries_df[iso3_col].astype(str).str.strip().str.len() == 3
) & (
    countries_df[iso3_col].astype(str).str.strip().str.match(r'^[A-Z]{3}$', na=False)
)
countries_df = countries_df[valid_mask].copy()
print(f"   After filtering (valid 3-letter ISO3): {len(countries_df)} unique country pairs")

if len(countries_df) == 0:
    print("   ⚠ WARNING: No valid ISO3 codes found after filtering!")
    print("   ⚠ Showing sample ISO3 values:")
    sample_iso3 = df_2025[iso3_col].dropna().head(10).tolist()
    for val in sample_iso3:
        print(f"      {repr(val)} (len: {len(str(val))})")

COUNTRIES = [(row[iso3_col], row[country_col]) for _, row in countries_df.iterrows()]

print(f"✓ Extracted {len(COUNTRIES)} unique countries from dataset")
if len(COUNTRIES) > 0:
    print(f"  First 5: {[c[1] for c in COUNTRIES[:5]]}")
    print(f"  Last 5: {[c[1] for c in COUNTRIES[-5:]]}")
else:
    print("  ⚠ WARNING: No valid countries extracted!")
    print("  ⚠ Check dataset format and ISO3 column")

# The 8 metrics to audit
METRICS = [
    'Total Training Compute (FLOP)',
    'National Hardware Compute Frontier (FLOP/s)',
    'Total Number of AI High-Level Publications',
    'Estimated Private AI Investment (USD)',
    'Total AI Patents Granted',
    'AI Workforce Size (Estimated)',
    'Government AI Readiness Index',
    'Specialized AI Infrastructure Score'
]

# ============================================================================
# VALIDATION - Verify count before generating
# ============================================================================
print("\n" + "="*70)
print("VALIDATION - Verifying query count")
print("="*70)

expected_countries = 227
expected_queries = expected_countries * len(METRICS)  # 227 × 8 = 1,816

if len(COUNTRIES) != expected_countries:
    print(f"⚠ WARNING: Found {len(COUNTRIES)} countries, expected {expected_countries}")
    print(f"  This will generate {len(COUNTRIES) * len(METRICS):,} queries instead of 1,816")
    if can_track_skipped and rows_skipped:
        print(f"  ⚠ This discrepancy may be due to {len(rows_skipped)} skipped rows during CSV parsing")
        print(f"  ⚠ Check if the skipped rows contained 2025 data for missing countries")
    elif not can_track_skipped:
        print(f"  ⚠ Note: Some rows may have been skipped during parsing (count not available)")
        print(f"  ⚠ This may explain the country count discrepancy")
else:
    print(f"✓ Country count correct: {len(COUNTRIES)}")
    if can_track_skipped and rows_skipped:
        print(f"  ℹ Note: {len(rows_skipped)} rows were skipped, but country count is still correct")

if len(COUNTRIES) * len(METRICS) != expected_queries:
    print(f"⚠ WARNING: Will generate {len(COUNTRIES) * len(METRICS):,} queries, expected {expected_queries:,}")
else:
    print(f"✓ Query count will be: {expected_queries:,} (227 × 8)")

# ============================================================================
# GENERATE QUERIES
# ============================================================================
print("\n" + "="*70)
print("GENERATING AUDIT QUERIES")
print("="*70)
print(f"Countries: {len(COUNTRIES)}")
print(f"Metrics: {len(METRICS)}")
print(f"Total queries: {len(COUNTRIES) * len(METRICS):,}")

# Generate queries
audit_queries = []
for iso3, country in COUNTRIES:
    for metric in METRICS:
        query = f"What is the {metric} for {country} in 2025?"
        audit_queries.append({
            'ISO3': iso3,
            'Country': country,
            'Metric': metric,
            'Query': query,
            'Ground_Truth': None
        })

# Validation: Print count before writing
print(f"\n✓ Generated {len(audit_queries):,} queries")
if len(audit_queries) == expected_queries:
    print(f"✓ ✅ COUNT VERIFIED: Exactly {expected_queries:,} queries (227 × 8)")
else:
    print(f"⚠ Count is {len(audit_queries):,}, expected {expected_queries:,}")

# Save to file using %%writefile
print("\nSaving to audit_data.py...")
print("="*70)

# Create file content
file_content = """# ============================================================================
# AUDIT DATA - Generated Programmatically
# ============================================================================
# This file contains 1,816 audit queries (227 countries × 8 metrics)
# Generated automatically - do not edit manually
# ============================================================================

audit_queries = [
"""

# Add each query
for query in audit_queries:
    file_content += "    {\n"
    file_content += f"        'ISO3': {repr(query['ISO3'])},\n"
    file_content += f"        'Country': {repr(query['Country'])},\n"
    file_content += f"        'Metric': {repr(query['Metric'])},\n"
    file_content += f"        'Query': {repr(query['Query'])},\n"
    file_content += f"        'Ground_Truth': {repr(query['Ground_Truth'])},\n"
    file_content += "    },\n"

file_content += "]\n"

# Write file
with open('audit_data.py', 'w', encoding='utf-8') as f:
    f.write(file_content)

print(f"✓ Saved {len(audit_queries):,} queries to audit_data.py")
import os
file_size = os.path.getsize('audit_data.py') / 1024
print(f"✓ File size: {file_size:.2f} KB")

# Verify
import sys
import importlib
importlib.invalidate_caches()
if 'audit_data' in sys.modules:
    del sys.modules['audit_data']

import audit_data
importlib.reload(audit_data)
verify_count = len(audit_data.audit_queries)
print(f"\n✓ Verification: {verify_count:,} queries loaded from audit_data.py")

# Self-check: Verify count
print("\n" + "="*70)
print("✅ GENERATION COMPLETE")
print("="*70)
print(f"✓ Generated queries: {len(audit_queries):,}")
print(f"✓ Verified in file: {verify_count:,}")
print(f"✓ Expected: 1,816 (227 countries × 8 metrics)")

if len(audit_queries) == 1816 and verify_count == 1816:
    print("✓ ✅ COUNT VERIFIED: All 1,816 queries generated successfully!")
elif len(audit_queries) == verify_count:
    print(f"⚠ Count matches but is {len(audit_queries):,} (expected 1,816)")
else:
    print(f"⚠ Mismatch: Generated {len(audit_queries):,} but file has {verify_count:,}")

print("✓ Ready to run the audit script!")
print("="*70)
