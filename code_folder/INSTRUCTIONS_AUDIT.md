# Census Audit - Setup Instructions

## Step 1: Create Data File

Run this cell FIRST (with `%%writefile` magic command):

```python
%%writefile audit_data.py

# ============================================================================
# AUDIT DATA - Paste your 1,816 queries here
# ============================================================================

audit_queries = [
    # PASTE YOUR FULL 1,816 QUERIES HERE
    # Format: {'ISO3': '...', 'Country': '...', 'Metric': '...', 'Query': '...'},
    # Make sure there's no trailing comma after the last item
]
```

**Important:**
- Replace the comment with your actual 1,816 queries
- Ensure proper Python list syntax (no trailing comma)
- Each query should be a dictionary with: ISO3, Country, Metric, Query (and optionally Ground_Truth)

## Step 2: Run Audit Script

After Step 1 completes, run the main audit cell (`colab_audit_census.py`).

The script will:
- Import `audit_queries` from `audit_data.py`
- Check for existing results and resume if needed
- Process all queries with incremental saving
- Show progress every 10 queries

## Troubleshooting

If you get `ModuleNotFoundError: No module named 'audit_data'`:
- Make sure you ran the `%%writefile` cell first
- Check that `audit_data.py` exists in the same directory

If you get syntax errors:
- Check for trailing commas in your list
- Verify all quotes are properly closed
- Ensure all dictionaries have the required keys
