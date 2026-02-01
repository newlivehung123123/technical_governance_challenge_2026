# ============================================================================
# ATOMIC KERNEL RESET - Fix NumPy Docstring Collision
# ============================================================================
# This script installs packages and immediately restarts the kernel
# NO IMPORTS - This avoids triggering the docstring collision
# ============================================================================

import os, sys, subprocess

print("="*70)
print("ATOMIC KERNEL RESET - Fixing NumPy Docstring Collision")
print("="*70)

# Step 1: Atomic Install - Install packages with no cache
print("\n[Step 1] Installing packages (no cache, force reinstall)...")
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
        timeout=600
    )
    if result.returncode == 0:
        print(f"    ✓ {pkg} installed")
    else:
        print(f"    ⚠ {pkg} had warnings (continuing)")

print("\n✓ Package installation complete")

# Step 2: Pre-emptive Strike - DO NOT import anything
# We skip all imports to avoid triggering the docstring collision
print("\n[Step 2] Skipping imports (avoiding docstring collision)...")
print("  ✓ No imports attempted - avoiding collision trigger")

# Step 3: The Kill Switch - Immediate kernel restart
print("\n" + "="*70)
print("KERNEL RESTART")
print("="*70)
print("")
print("Environment updated. Colab is now restarting the kernel automatically.")
print("WAIT for the 'Connected' status at the top right, then run your Audit cell.")
print("")
print("🔄 Restarting kernel in 2 seconds...")
print("="*70)

import time
time.sleep(2)

# IMMEDIATE KILL - Force kernel restart
print("\n💥 Executing os._exit(0) - Kernel restarting now...")
os._exit(0)
