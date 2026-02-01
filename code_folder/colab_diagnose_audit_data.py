# ============================================================================
# DIAGNOSTIC: Inspect audit_data.py and Fix if Empty
# ============================================================================
# This cell checks if audit_data.py contains your queries
# ============================================================================

import os
import sys
import importlib
from pathlib import Path

print("="*70)
print("DIAGNOSTIC: Checking audit_data.py")
print("="*70)

# Step 1: Content Inspection
print("\n[Step 1] Content Inspection - First 20 lines:")
print("-" * 70)
try:
    with open('audit_data.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()[:20]
        if not lines:
            print("  ⚠ File is empty")
        else:
            for i, line in enumerate(lines, 1):
                # Show line number and content (truncate long lines)
                content = line.rstrip()[:80]
                if len(line.rstrip()) > 80:
                    content += "..."
                print(f"  {i:2d}: {content}")
            if len(lines) < 5:
                print("  ⚠ File is very short - likely incomplete")
except FileNotFoundError:
    print("  ✗ File audit_data.py does not exist!")
except Exception as e:
    print(f"  ✗ Error reading file: {e}")

# Step 2: Size Verification
print("\n[Step 2] Size Verification:")
print("-" * 70)
try:
    file_path = Path("audit_data.py")
    if file_path.exists():
        size_bytes = os.path.getsize("audit_data.py")
        size_kb = size_bytes / 1024
        print(f"  File size: {size_bytes:,} bytes ({size_kb:.2f} KB)")
        
        if size_bytes < 100:
            print("  ⚠ File is very small - likely empty or incomplete")
        elif size_kb < 50:
            print("  ⚠ File is smaller than expected for 1,816 queries")
        else:
            print("  ✓ File size looks reasonable")
    else:
        print("  ✗ File does not exist!")
except Exception as e:
    print(f"  ✗ Error checking file size: {e}")

# Step 3: Try to Import and Check Count
print("\n[Step 3] Import and Count Check:")
print("-" * 70)
try:
    # Force reload
    importlib.invalidate_caches()
    if 'audit_data' in sys.modules:
        del sys.modules['audit_data']
    
    import audit_data
    importlib.reload(audit_data)
    query_count = len(audit_data.audit_queries)
    
    print(f"  ✓ Successfully imported audit_data")
    print(f"  ✓ Query count: {query_count:,}")
    
    if query_count == 0:
        print("\n" + "="*70)
        print("❌ PROBLEM DETECTED: audit_data.py is empty or has no queries")
        print("="*70)
        print("\n📋 MANUAL OVERRIDE INSTRUCTIONS:")
        print("="*70)
        print("\n1. DELETE the existing audit_data.py:")
        print("   - Click the folder icon in Colab sidebar")
        print("   - Right-click on 'audit_data.py'")
        print("   - Select 'Delete'")
        print("\n2. CREATE a new cell and run this code:")
        print("="*70)
        print("""
%%writefile audit_data.py

# ============================================================================
# AUDIT DATA - 1,816 Census Queries
# ============================================================================

audit_queries = [
    # PASTE YOUR 1,816 QUERIES HERE
    # Format: {'ISO3': '...', 'Country': '...', 'Metric': '...', 'Query': '...', 'Ground_Truth': ...}
    # Make sure there's NO trailing comma after the last item
]
""")
        print("="*70)
        print("\n3. PASTE your 1,816 queries between the brackets")
        print("\n4. RUN the cell and wait for 'Writing audit_data.py' message")
        print("\n5. RUN this diagnostic cell again to verify")
        print("="*70)
    elif query_count < 100:
        print(f"\n  ⚠ WARNING: Only {query_count} queries found. Expected ~1,816.")
        print("  ⚠ Your file may be incomplete. Check audit_data.py")
    else:
        print(f"\n  ✅ SUCCESS: Found {query_count:,} queries!")
        print("  ✅ audit_data.py is ready to use")
        
except ImportError as e:
    print(f"  ✗ Could not import audit_data: {e}")
    print("\n" + "="*70)
    print("❌ FILE NOT FOUND")
    print("="*70)
    print("\n📋 SOLUTION:")
    print("="*70)
    print("\nCreate audit_data.py using this cell:")
    print("="*70)
    print("""
%%writefile audit_data.py

audit_queries = [
    # PASTE YOUR 1,816 QUERIES HERE
]
""")
    print("="*70)
except Exception as e:
    print(f"  ✗ Error: {e}")
    print("\n  ⚠ There may be a syntax error in audit_data.py")
    print("  ⚠ Check the file for syntax issues")

# Step 4: Success Check (if queries found)
print("\n" + "="*70)
if 'query_count' in locals() and query_count > 0:
    print("✅ DIAGNOSTIC COMPLETE")
    print("="*70)
    print(f"✓ audit_data.py contains {query_count:,} queries")
    print("✓ Ready to run the audit script")
else:
    print("⚠️  DIAGNOSTIC COMPLETE - ACTION REQUIRED")
    print("="*70)
    print("⚠️  Follow the instructions above to fix audit_data.py")
print("="*70)
