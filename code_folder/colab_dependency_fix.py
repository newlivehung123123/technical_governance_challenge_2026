"""
Google Colab Dependency Fix Cell
=================================
Run this in a single cell to fix NumPy 2.x binary mismatch and dependency conflicts.

This cell:
1. Purges corrupted ghost folders in dist-packages
2. Disables debugger crash loops
3. Surgically installs compatible package versions
4. Refreshes Python path without kernel restart
"""

import os
import sys
import shutil
import site
import subprocess
from pathlib import Path

print("=" * 70)
print("GOOGLE COLAB DEPENDENCY FIX")
print("=" * 70)

# Step 1: Disable debugger validation to stop crash loops
print("\n[Step 1] Disabling debugger validation...")
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
print("✓ Debugger validation disabled")

# Step 2: Purge corrupted ghost folders in dist-packages
print("\n[Step 2] Purging corrupted ghost folders...")
dist_packages = Path("/usr/local/lib/python3.12/dist-packages")
if dist_packages.exists():
    ghost_folders = [d for d in dist_packages.iterdir() 
                     if d.is_dir() and d.name.startswith('~')]
    
    for ghost in ghost_folders:
        try:
            print(f"  Removing: {ghost.name}")
            shutil.rmtree(ghost, ignore_errors=True)
        except Exception as e:
            print(f"  Warning: Could not remove {ghost.name}: {e}")
    
    print(f"✓ Purged {len(ghost_folders)} ghost folders")
else:
    print("  Note: dist-packages path not found (may be different Python version)")

# Step 3: Force uninstall conflicting packages
print("\n[Step 3] Force uninstalling conflicting packages...")
packages_to_remove = ['numpy', 'pandas', 'transformers', 'tokenizers', 'bitsandbytes']
for pkg in packages_to_remove:
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', pkg], 
                      capture_output=True, check=False)
    except:
        pass
print("✓ Cleaned conflicting packages")

# Step 4: Install exact versions with --no-deps to avoid conflicts
print("\n[Step 4] Installing compatible versions (surgical install)...")

# Define exact versions for compatibility
PACKAGE_VERSIONS = {
    'numpy': '2.1.0',
    'pandas': '2.2.2',
    'transformers': '4.44.2',
    'tokenizers': '0.20.0',
    'bitsandbytes': '0.43.3'
}

# Install order matters: numpy first (foundation), then others
install_order = ['numpy', 'pandas', 'tokenizers', 'transformers', 'bitsandbytes']

for pkg in install_order:
    version = PACKAGE_VERSIONS[pkg]
    print(f"  Installing {pkg}=={version}...")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--no-deps', 
             f'{pkg}=={version}', '--force-reinstall', '--no-cache-dir'],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"    ✓ {pkg} installed")
    except subprocess.CalledProcessError as e:
        print(f"    ✗ Error installing {pkg}: {e.stderr}")
        # Try with --upgrade flag as fallback
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--no-deps', 
                 f'{pkg}=={version}', '--upgrade', '--force-reinstall', '--no-cache-dir'],
                capture_output=True,
                check=True
            )
            print(f"    ✓ {pkg} installed (with --upgrade)")
        except:
            print(f"    ✗ Failed to install {pkg}")

# Step 5: Install critical dependencies that bitsandbytes needs
print("\n[Step 5] Installing critical dependencies...")
critical_deps = [
    'accelerate>=0.24.0',
    'safetensors>=0.3.0',
    'packaging>=21.0'
]

for dep in critical_deps:
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', dep, '--upgrade', '--no-cache-dir'],
            capture_output=True,
            check=False
        )
        print(f"  ✓ Installed {dep}")
    except:
        print(f"  ⚠ Could not install {dep}")

# Step 6: Refresh Python path and reload modules
print("\n[Step 6] Refreshing Python path...")

# Clear module cache to force reload
modules_to_clear = ['numpy', 'pandas', 'transformers', 'tokenizers']
for mod in modules_to_clear:
    if mod in sys.modules:
        del sys.modules[mod]

# Refresh site-packages
site_packages = [p for p in site.getsitepackages() if 'site-packages' in p or 'dist-packages' in p]
for sp in site_packages:
    if Path(sp).exists():
        site.addsitedir(sp)

# Force reload importlib
import importlib
importlib.invalidate_caches()

print("✓ Python path refreshed")

# Step 7: Verify imports work
print("\n[Step 7] Verifying imports...")
try:
    import numpy as np
    print(f"  ✓ numpy {np.__version__} imported successfully")
    print(f"    NumPy dtype size: {np.dtype('float64').itemsize} bytes")
except Exception as e:
    print(f"  ✗ numpy import failed: {e}")

try:
    import pandas as pd
    print(f"  ✓ pandas {pd.__version__} imported successfully")
except Exception as e:
    print(f"  ✗ pandas import failed: {e}")

try:
    import transformers
    print(f"  ✓ transformers {transformers.__version__} imported successfully")
except Exception as e:
    print(f"  ✗ transformers import failed: {e}")

try:
    import tokenizers
    print(f"  ✓ tokenizers {tokenizers.__version__} imported successfully")
except Exception as e:
    print(f"  ✗ tokenizers import failed: {e}")

try:
    import bitsandbytes
    print(f"  ✓ bitsandbytes imported successfully")
except Exception as e:
    print(f"  ⚠ bitsandbytes import failed (may need CUDA): {e}")

print("\n" + "=" * 70)
print("DEPENDENCY FIX COMPLETE")
print("=" * 70)
print("\nYou can now import numpy and transformers in the next cell.")
print("If you still see errors, restart the runtime once (not multiple times).")
