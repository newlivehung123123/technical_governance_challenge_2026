# ============================================================================
# CENSUS AUDIT ANALYSIS - Comprehensive Analysis of AI Knowledge vs. Refusal
# ============================================================================
# Analyzes census_audit_results.csv to understand AI knowledge patterns
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from collections import Counter

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("="*70)
print("CENSUS AUDIT ANALYSIS")
print("="*70)

# ============================================================================
# DATA LOADING
# ============================================================================
print("\n[Step 1] Loading data files...")

# Load audit results
audit_results_paths = [
    Path("/content/drive/MyDrive/census_audit_results.csv"),
    Path("census_audit_results.csv"),
    Path("/content/census_audit_results.csv"),
]

audit_path = None
for path in audit_results_paths:
    if path.exists():
        audit_path = path
        print(f"  ✓ Found audit results: {audit_path}")
        break

if audit_path is None:
    raise FileNotFoundError("census_audit_results.csv not found!")

df_audit = pd.read_csv(audit_path, encoding='utf-8')
print(f"  ✓ Loaded {len(df_audit):,} audit results")

# Load original dataset
original_paths = [
    Path("GAID_MASTER_V2_COMPILATION_FINAL.csv"),
    Path("GAID_w1_v2_dataset/GAID_MASTER_V2_COMPILATION_FINAL.csv"),
    Path("../GAID_w1_v2_dataset/GAID_MASTER_V2_COMPILATION_FINAL.csv"),
    Path("/content/GAID_MASTER_V2_COMPILATION_FINAL.csv"),
    Path("/content/drive/MyDrive/GAID_MASTER_V2_COMPILATION_FINAL.csv"),
]

original_path = None
for path in original_paths:
    if path.exists():
        original_path = path
        print(f"  ✓ Found original dataset: {original_path}")
        break

if original_path is None:
    print("  ⚠ Original dataset not found - will skip hallucination check")
    df_original = None
else:
    try:
        df_original = pd.read_csv(original_path, encoding='utf-8', on_bad_lines='skip', engine='python')
        print(f"  ✓ Loaded {len(df_original):,} rows from original dataset")
    except Exception as e:
        print(f"  ⚠ Could not load original dataset: {e}")
        df_original = None

# ============================================================================
# RESPONSE CATEGORIZATION
# ============================================================================
print("\n[Step 2] Categorizing AI responses...")

def categorize_response(response_text):
    """
    Categorize AI response into:
    - "Provides Number/Fact": Contains numeric data or substantial context
    - "Admits No Data/Refusal": Explicitly states no data/doesn't know
    - "Misunderstanding/Correction": Misunderstands the question
    
    This matches the logic from CensusDashboard.jsx for consistency.
    
    STEP 1: Check for explicit Refusal Patterns FIRST (before numbers)
    STEP 2: Check for Misunderstanding/Correction patterns
    STEP 3: Perform Sanitized Number Check (exclude years, only technical data)
    """
    if pd.isna(response_text) or response_text == "":
        return "Empty Response"
    
    response_lower = str(response_text).lower()
    response_original = str(response_text)
    
    # STEP 1: Check for explicit Refusal Patterns FIRST (before checking numbers)
    # Comprehensive refusal patterns matching CensusDashboard.jsx
    refusal_patterns = [
        r"i don'?t have access",
        r"i don'?t have (the|any|specific) data",
        r"no specific data",
        r"unavailable",
        r"not available",
        r"i couldn'?t find",
        r"i can'?t find",
        r"i don'?t know",
        r"no data",
        r"no information",
        r"not publicly available",
        r"not readily available",
        r"limited.*data",
        r"i think there may be",
        r"confusion",
        r"misunderstanding",
        r"not a widely recognized",
        r"doesn'?t exist",
        r"not possible",
        r"not typically",
        r"not usually",
        r"i'm not able to",
        r"i cannot provide",
        r"i can'?t provide",
        r"unable to",
        r"lack of.*data",
        r"absence of.*data",
        r"no reliable sources",
        r"no reliable data",
        r"couldn'?t find any",
        r"can'?t find any",
        r"don'?t have access to",
        r"don'?t have information",
        r"not in my",
        r"not part of",
        r"outside.*scope",
        r"beyond.*knowledge",
        r"beyond my knowledge",
        r"don'?t have (access to|information|data)",
        r"no (data|information|specific data|available data)",
        r"unable to (provide|access|find|retrieve)",
        r"cannot (provide|access|find|retrieve|determine)",
        r"do not have",
        r"lack of (data|information)",
        r"no (reliable|specific|exact|precise) (data|information)",
        r"i (don'?t|do not) (have|know|possess)",
        r"i'm (unable|not able)",
        r"i cannot",
        r"there is no (data|information)",
        r"no (such|specific) (metric|data|information)",
    ]
    
    # STEP 1: Check for explicit refusal FIRST - return immediately if found
    for pattern in refusal_patterns:
        if re.search(pattern, response_lower):
            return "Admits No Data/Refusal"
    
    # STEP 2: Check for misunderstanding/correction patterns
    misunderstanding_patterns = [
        r"no such (metric|concept|measure)",
        r"that (metric|concept|measure) (doesn'?t|does not) (exist|apply)",
        r"not a (valid|real|standard) (metric|measure)",
        r"i (don'?t|do not) understand",
        r"could you (clarify|rephrase|explain)",
        r"unclear (what|which)",
        r"ambiguous",
        r"not sure what you mean",
    ]
    
    for pattern in misunderstanding_patterns:
        if re.search(pattern, response_lower):
            return "Misunderstanding/Correction"
    
    # STEP 3: Sanitized Number Check - only count technical data, exclude years 2021-2025
    # Years to exclude (common model-generated noise)
    excluded_years = ['2021', '2022', '2023', '2024', '2025']
    
    # Remove excluded years from the text for number checking
    response_sanitized = response_original
    for year in excluded_years:
        response_sanitized = response_sanitized.replace(year, '')
    
    # Check for scientific notation (always valid technical data)
    has_scientific = bool(re.search(r'\d+\.?\d*[eE][+-]?\d+', response_original))
    
    # Check for percentages (always valid)
    has_percentage = bool(re.search(r'\d+\.?\d*%', response_original))
    
    # Check for numbers with technical units (FLOP, USD, percent, etc.)
    technical_unit_patterns = [
        r'\d+[.,]?\d*\s*(flop|flops|usd|\$|dollar|dollars|percent|%|million|billion|trillion|thousand|k|m|b|t)',
        r'\d+[.,]?\d*\s*(gpu|cpu|core|cores|parameter|parameters|model|models)',
        r'\d+[.,]?\d*\s*(watt|watts|ghz|mhz|gb|tb|mb|petabyte|exabyte)',
    ]
    has_technical_unit = any(re.search(pattern, response_lower) for pattern in technical_unit_patterns)
    
    # Check for decimal numbers (likely technical measurements, not years)
    # But exclude if the only numbers are years
    has_decimal = bool(re.search(r'\d+\.\d+', response_sanitized))
    
    # Check for large non-date integers (>= 1000, not in excluded years)
    # Look for numbers that are clearly technical (with context or very large)
    large_number_matches = re.finditer(r'\b([1-9]\d{2,})\b', response_sanitized)
    has_large_technical_number = False
    
    for match in large_number_matches:
        num_str = match.group(1)
        try:
            num_val = int(num_str.replace(',', ''))
            # Accept numbers >= 1000 (likely technical, not dates)
            if num_val >= 1000:
                # Check if followed by unit or in technical context
                match_end = match.end()
                context_after = response_lower[match_end:match_end + 15]
                # If followed by unit or technical term, it's valid
                if any(term in context_after for term in ['flop', 'usd', '$', 'percent', '%', 'million', 'billion', 'parameter', 'model', 'gpu', 'cpu']):
                    has_large_technical_number = True
                    break
                # Very large numbers (>= 10000) are likely technical
                if num_val >= 10000:
                    has_large_technical_number = True
                    break
        except:
            continue
    
    # If has valid technical numbers (scientific, percentage, units, decimals, or large technical numbers), it's factual
    if has_scientific or has_percentage or has_technical_unit or has_decimal or has_large_technical_number:
        return "Provides Number/Fact"
    
    # If no numbers and no explicit refusal, might be vague response
    if len(response_text) < 50:
        return "Vague/Short Response"
    
    return "Other Response"

# Apply categorization
df_audit['Response_Category'] = df_audit['AI_Response'].apply(categorize_response)

# Count categories
category_counts = df_audit['Response_Category'].value_counts()
print("\n  Response Categories:")
for category, count in category_counts.items():
    pct = (count / len(df_audit)) * 100
    print(f"    {category}: {count:,} ({pct:.1f}%)")

# ============================================================================
# GEOGRAPHIC ANALYSIS
# ============================================================================
print("\n[Step 3] Geographic Analysis...")

# Calculate knowledge rate per country
country_stats = df_audit.groupby('Country').agg({
    'Response_Category': lambda x: (x == 'Provides Number/Fact').sum(),
    'ISO3': 'count'
}).rename(columns={'Response_Category': 'Factual_Count', 'ISO3': 'Total_Queries'})

country_stats['Knowledge_Rate'] = (country_stats['Factual_Count'] / country_stats['Total_Queries']) * 100
country_stats['Refusal_Rate'] = df_audit.groupby('Country')['Response_Category'].apply(
    lambda x: (x == 'Admits No Data/Refusal').sum() / len(x) * 100
)

# Keep unsorted for now - we'll sort by the metric we need for each visualization
# Don't sort here to avoid affecting other calculations

print("\n  Top 10 Countries by Knowledge Rate (Factual Answers):")
top_knowledge_print = country_stats.nlargest(10, 'Knowledge_Rate')
print(top_knowledge_print[['Factual_Count', 'Total_Queries', 'Knowledge_Rate']].to_string())

print("\n  Top 10 Countries by Refusal Rate (Most Ignorance):")
top_refusals_print = country_stats.nlargest(10, 'Refusal_Rate')
print(top_refusals_print[['Factual_Count', 'Total_Queries', 'Refusal_Rate']].to_string())

# ============================================================================
# METRIC SENSITIVITY ANALYSIS
# ============================================================================
print("\n[Step 4] Metric Sensitivity Analysis...")

metric_stats = df_audit.groupby('Metric').agg({
    'Response_Category': lambda x: (x == 'Provides Number/Fact').sum(),
    'ISO3': 'count'
}).rename(columns={'Response_Category': 'Factual_Count', 'ISO3': 'Total_Queries'})

metric_stats['Knowledge_Rate'] = (metric_stats['Factual_Count'] / metric_stats['Total_Queries']) * 100
metric_stats['Refusal_Rate'] = df_audit.groupby('Metric')['Response_Category'].apply(
    lambda x: (x == 'Admits No Data/Refusal').sum() / len(x) * 100
)

metric_stats = metric_stats.sort_values('Knowledge_Rate', ascending=True)

print("\n  Metrics Ranked by Difficulty (Lowest Knowledge Rate = Most Difficult):")
print(metric_stats[['Factual_Count', 'Total_Queries', 'Knowledge_Rate', 'Refusal_Rate']].to_string())

# ============================================================================
# HALLUCINATION CHECK (if original data available)
# ============================================================================
if df_original is not None:
    print("\n[Step 5] Hallucination Check...")
    
    # Try to extract numeric values from AI responses
    def extract_number(text):
        """Extract first significant number from text, handling text-based multipliers"""
        if pd.isna(text):
            return None
        
        text_str = str(text).lower()
        
        # Multiplier dictionary
        multipliers = {
            'trillion': 1e12,
            'billion': 1e9,
            'million': 1e6,
            'thousand': 1e3,
            'k': 1e3,
            'm': 1e6,
            'b': 1e9,
            't': 1e12,
        }
        
        # Look for numbers (including scientific notation)
        patterns = [
            r'(\d+\.?\d*[eE][+-]?\d+)',  # Scientific notation
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  # Regular numbers with commas
            r'(\d+\.?\d+)',  # Decimal numbers
            r'(\d+)',  # Integers
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_str)
            if match:
                num_str = match.group(1).replace(',', '')
                try:
                    number = float(num_str)
                    
                    # Check for text-based multipliers after the number
                    text_after = text_str[match.end():match.end()+20].strip()
                    for mult_text, mult_value in multipliers.items():
                        if text_after.startswith(mult_text):
                            number *= mult_value
                            break
                    
                    return number
                except:
                    continue
        return None
    
    # Extract numbers from AI responses
    df_audit['AI_Number'] = df_audit['AI_Response'].apply(extract_number)
    
    # Try to match with original data
    # This is approximate - you may need to adjust based on your data structure
    if 'Ground_Truth' in df_audit.columns:
        # Try to extract ground truth numbers
        df_audit['Ground_Truth_Number'] = df_audit['Ground_Truth'].apply(extract_number)
        
        # Calculate error for responses with both numbers
        valid_comparisons = df_audit.dropna(subset=['AI_Number', 'Ground_Truth_Number'])
        
        if len(valid_comparisons) > 0:
            # Calculate relative error
            valid_comparisons = valid_comparisons.copy()
            valid_comparisons['Relative_Error'] = np.abs(
                (valid_comparisons['AI_Number'] - valid_comparisons['Ground_Truth_Number']) / 
                (valid_comparisons['Ground_Truth_Number'] + 1e-10)
            ) * 100
            
            print(f"  ✓ Found {len(valid_comparisons):,} responses with comparable numbers")
            print(f"  Average Relative Error: {valid_comparisons['Relative_Error'].mean():.2f}%")
            print(f"  Median Relative Error: {valid_comparisons['Relative_Error'].median():.2f}%")
            
            # Flag high errors (>50% relative error as potential hallucination)
            high_error = valid_comparisons[valid_comparisons['Relative_Error'] > 50]
            print(f"  ⚠ Potential Hallucinations (>50% error): {len(high_error):,} ({len(high_error)/len(valid_comparisons)*100:.1f}%)")
        else:
            print("  ⚠ Could not extract comparable numbers from responses")
    else:
        print("  ⚠ Ground_Truth column not found in audit results")
else:
    print("\n[Step 5] Skipping Hallucination Check (original dataset not available)")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[Step 6] Generating Visualizations...")

# Create output directory
output_dir = Path("/content/drive/MyDrive/audit_analysis") if Path("/content/drive/MyDrive").exists() else Path(".")
output_dir.mkdir(exist_ok=True)

# Visualization 1: Response Category Distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Pie chart
category_counts.plot(kind='pie', ax=axes[0], autopct='%1.1f%%', startangle=90)
axes[0].set_title('Distribution of AI Response Categories', fontsize=14, fontweight='bold')
axes[0].set_ylabel('')

# Bar chart
category_counts.plot(kind='bar', ax=axes[1], color='steelblue')
axes[1].set_title('Response Category Counts', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Category', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(output_dir / 'response_categories.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: response_categories.png")
plt.show()

# Visualization 2: Global Ignorance Map (Bar Chart by Country)
fig, ax = plt.subplots(figsize=(16, 10))

# CRITICAL FIX: Sort by Refusal_Rate in DESCENDING order (MOST refusals first)
# Get top 30 countries with HIGHEST refusal rates
top_refusals = country_stats.nlargest(30, 'Refusal_Rate')

# For barh, we need to reverse so highest appears at top (barh displays bottom to top)
# Sort ascending for display, then reverse the order
top_refusals_display = top_refusals.sort_values('Refusal_Rate', ascending=True)

bars = ax.barh(range(len(top_refusals_display)), top_refusals_display['Refusal_Rate'], color='coral')
ax.set_yticks(range(len(top_refusals_display)))
ax.set_yticklabels(top_refusals_display.index, fontsize=9)
ax.set_xlabel('Refusal Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Global Ignorance Map: Top 30 Countries by AI Refusal Rate\n(Where AI Most Often Admits No Data - Countries with MOST Refusals)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(top_refusals_display.iterrows()):
    ax.text(row['Refusal_Rate'] + 1, i, f"{row['Refusal_Rate']:.1f}%", 
            va='center', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'global_ignorance_map.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: global_ignorance_map.png")
plt.show()

# Visualization 3: Metric Difficulty Analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Knowledge rate by metric
metric_stats_sorted = metric_stats.sort_values('Knowledge_Rate', ascending=True)
bars1 = axes[0].barh(range(len(metric_stats_sorted)), metric_stats_sorted['Knowledge_Rate'], 
                     color='steelblue')
axes[0].set_yticks(range(len(metric_stats_sorted)))
axes[0].set_yticklabels([m[:40] + '...' if len(m) > 40 else m for m in metric_stats_sorted.index], 
                        fontsize=9)
axes[0].set_xlabel('Knowledge Rate (%)', fontsize=12, fontweight='bold')
axes[0].set_title('Metric Difficulty: Knowledge Rate by Metric\n(Lower = More Difficult)', 
                  fontsize=14, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Refusal rate by metric
bars2 = axes[1].barh(range(len(metric_stats_sorted)), metric_stats_sorted['Refusal_Rate'], 
                     color='coral')
axes[1].set_yticks(range(len(metric_stats_sorted)))
axes[1].set_yticklabels([m[:40] + '...' if len(m) > 40 else m for m in metric_stats_sorted.index], 
                        fontsize=9)
axes[1].set_xlabel('Refusal Rate (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Metric Difficulty: Refusal Rate by Metric\n(Higher = More Difficult)', 
                  fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'metric_difficulty.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: metric_difficulty.png")
plt.show()

# Visualization 4: Top Countries by Knowledge Rate
fig, ax = plt.subplots(figsize=(14, 8))

top_knowledge = country_stats.nlargest(20, 'Knowledge_Rate')
# Sort ascending for display (barh shows bottom to top)
top_knowledge_display = top_knowledge.sort_values('Knowledge_Rate', ascending=True)
bars = ax.barh(range(len(top_knowledge_display)), top_knowledge_display['Knowledge_Rate'], color='green')
ax.set_yticks(range(len(top_knowledge_display)))
ax.set_yticklabels(top_knowledge_display.index, fontsize=10)
ax.set_xlabel('Knowledge Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Countries: Where AI Provides Most Factual Answers', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(top_knowledge_display.iterrows()):
    ax.text(row['Knowledge_Rate'] + 1, i, f"{row['Knowledge_Rate']:.1f}%", 
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'top_countries_knowledge.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: top_countries_knowledge.png")
plt.show()

# Visualization 5: Heatmap of Knowledge Rate by Country and Metric (sample)
print("\n  Generating heatmap (this may take a moment)...")

# Sample top 20 countries by knowledge rate and all metrics for heatmap
top_countries = country_stats.nlargest(20, 'Knowledge_Rate').index
heatmap_data = []

for country in top_countries:
    country_data = []
    for metric in metric_stats.index:
        subset = df_audit[(df_audit['Country'] == country) & (df_audit['Metric'] == metric)]
        if len(subset) > 0:
            knowledge_rate = (subset['Response_Category'] == 'Provides Number/Fact').sum() / len(subset) * 100
        else:
            knowledge_rate = 0
        country_data.append(knowledge_rate)
    heatmap_data.append(country_data)

heatmap_df = pd.DataFrame(heatmap_data, index=top_countries, columns=metric_stats.index)

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(heatmap_df, annot=True, fmt='.0f', cmap='RdYlGn', 
            cbar_kws={'label': 'Knowledge Rate (%)'}, ax=ax, vmin=0, vmax=100)
ax.set_title('Knowledge Rate Heatmap: Top 20 Countries × All Metrics', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Country', fontsize=12)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'knowledge_heatmap.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: knowledge_heatmap.png")
plt.show()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

total_responses = len(df_audit)
factual_count = (df_audit['Response_Category'] == 'Provides Number/Fact').sum()
refusal_count = (df_audit['Response_Category'] == 'Admits No Data/Refusal').sum()
misunderstanding_count = (df_audit['Response_Category'] == 'Misunderstanding/Correction').sum()

print(f"\nTotal Responses Analyzed: {total_responses:,}")
print(f"Factual Responses: {factual_count:,} ({factual_count/total_responses*100:.1f}%)")
print(f"Refusals (No Data): {refusal_count:,} ({refusal_count/total_responses*100:.1f}%)")
print(f"Misunderstandings: {misunderstanding_count:,} ({misunderstanding_count/total_responses*100:.1f}%)")

best_known = country_stats.nlargest(1, 'Knowledge_Rate')
worst_known = country_stats.nsmallest(1, 'Knowledge_Rate')
most_refusals = country_stats.nlargest(1, 'Refusal_Rate')

print(f"\nBest Known Country: {best_known.index[0]} ({best_known.iloc[0]['Knowledge_Rate']:.1f}% factual)")
print(f"Least Known Country: {worst_known.index[0]} ({worst_known.iloc[0]['Knowledge_Rate']:.1f}% factual)")
print(f"Most Refusals: {most_refusals.index[0]} ({most_refusals.iloc[0]['Refusal_Rate']:.1f}% refusals)")

print(f"\nEasiest Metric: {metric_stats.index[-1]} ({metric_stats.iloc[-1]['Knowledge_Rate']:.1f}% factual)")
print(f"Hardest Metric: {metric_stats.index[0]} ({metric_stats.iloc[0]['Knowledge_Rate']:.1f}% factual)")

print("\n" + "="*70)
print("✅ Analysis Complete!")
print(f"📊 Visualizations saved to: {output_dir}")
print("="*70)
