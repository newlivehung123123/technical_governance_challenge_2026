# ============================================================================
# GOOGLE COLAB KERNEL RESET - FORCE KERNEL RESTART TO FIX DOCSTRING COLLISION
# ============================================================================
# This cell forces a kernel restart to flush empty_like docstring metadata
# Run this cell ONCE, wait for kernel restart, then run your main code
# ============================================================================

import os, sys, subprocess

print("="*70)
print("KERNEL RESET SCRIPT - Fixing empty_like Docstring Collision")
print("="*70)

# Step 1: Final Cleanup - Force reinstall with no cache
print("\n[Step 1] Final cleanup - Force reinstalling packages...")
packages = [
    'numpy==2.1.3',
    'transformers==4.46.0',
    'tokenizers==0.20.3',
    'bitsandbytes==0.43.3'
]

for pkg in packages:
    print(f"  Installing {pkg}...")
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall',
         '--no-cache-dir', pkg],
        capture_output=True,
        text=True,
        check=False,
        timeout=120
    )
    if result.returncode == 0:
        print(f"    ✓ {pkg} installed")
    else:
        print(f"    ⚠ {pkg} had issues (continuing anyway)")

print("\n✓ Final cleanup complete")

# Step 2: Verification Check - Try to import numpy
print("\n[Step 2] Verification check - Testing numpy import...")
import_failed = False
docstring_collision = False

try:
    import numpy as np
    print(f"  ✓ numpy imported successfully")
    print(f"  Version: {np.__version__}")
    
    # Try to use empty_like to check for docstring collision
    try:
        test_array = np.empty_like([1, 2, 3])
        print(f"  ✓ numpy.empty_like works (no docstring collision)")
    except RuntimeError as e:
        if 'empty_like' in str(e) or 'docstring' in str(e):
            print(f"  ✗ DOCSTRING COLLISION DETECTED: {e}")
            docstring_collision = True
            import_failed = True
        else:
            raise
except RuntimeError as e:
    if 'empty_like' in str(e) or 'docstring' in str(e):
        print(f"  ✗ DOCSTRING COLLISION DETECTED: {e}")
        docstring_collision = True
        import_failed = True
    else:
        print(f"  ✗ numpy import failed: {e}")
        import_failed = True
except Exception as e:
    print(f"  ✗ numpy import failed: {e}")
    import_failed = True

# Step 3: Hard Kill - Force kernel restart
# ALWAYS restart if import failed OR if this is first run
print("\n" + "="*70)

# Check if numpy is already cleanly imported (not first run)
numpy_already_clean = 'numpy' in sys.modules and not import_failed and not docstring_collision

# Restart if: import failed, docstring collision, OR first run
should_restart = import_failed or docstring_collision or not numpy_already_clean

if should_restart:
    print("⚠️  KERNEL RESTART REQUIRED")
    print("="*70)
    print("")
    print("⚠️ RESTARTING KERNEL TO APPLY FIX... WAIT 5 SECONDS THEN RUN THE NEXT CELL.")
    print("")
    print("📋 INSTRUCTIONS:")
    print("   1. Wait 5 seconds for the kernel to restart")
    print("   2. After restart, run the NEXT CELL (not this one again)")
    print("   3. Your environment should be clean after restart")
    print("   4. The 'Session crashed' notification is EXPECTED and NORMAL")
    print("")
    print("🔄 Restarting in 5 seconds...")
    print("="*70)
    
    import time
    time.sleep(5)
    
    # HARD KILL - Force kernel restart
    print("\n💥 EXECUTING os._exit(0) - Kernel restarting now...")
    import os
    os._exit(0)
else:
    print("✅ VERIFICATION PASSED")
    print("="*70)
    print("✓ numpy imported successfully")
    print("✓ No docstring collision detected")
    print("✓ Environment is clean - you can proceed!")
    print("="*70)
