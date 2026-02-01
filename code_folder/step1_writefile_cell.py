# ============================================================================
# STEP 1: CREATE DATA FILE
# ============================================================================
# Copy this entire cell into Google Colab and run it
# Replace the placeholder with your actual 1,816 queries
# ============================================================================

%%writefile audit_data.py

# ============================================================================
# AUDIT DATA - 1,816 Census Queries
# ============================================================================

audit_queries = [
    # PASTE YOUR 1,816 QUERIES HERE
    # Format: {'ISO3': '...', 'Country': '...', 'Metric': '...', 'Query': '...', 'Ground_Truth': ...}
    # Example:
    # {'ISO3': 'AFG', 'Country': 'Afghanistan', 'Metric': 'Total Training Compute (FLOP)', 'Query': 'What is the total training compute in FLOP for Afghanistan in 2025?', 'Ground_Truth': None},
    # {'ISO3': 'AFG', 'Country': 'Afghanistan', 'Metric': 'National Hardware Compute Frontier (FLOP/s)', 'Query': 'What is the national hardware compute frontier in FLOP/s for Afghanistan in 2025?', 'Ground_Truth': None},
    # ... [continue with all 1,816 queries]
    # IMPORTANT: No trailing comma after the last item!
]
