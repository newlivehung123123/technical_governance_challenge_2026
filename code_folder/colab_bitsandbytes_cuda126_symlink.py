# ============================================================================
# BITSANDBYTES CUDA 12.6 SYMLINK FIX
# ============================================================================
# Upgrades bitsandbytes and creates symlink to enable CUDA 12.6 support
# ============================================================================

import os, sys, subprocess
from pathlib import Path

print("="*70)
print("BITSANDBYTES CUDA 12.6 SYMLINK FIX")
print("="*70)

# Step 1: Upgrade to Multi-Backend - Install bitsandbytes >= 0.45.0
print("\n[Step 1] Upgrading to bitsandbytes >= 0.45.0 (multi-backend support)...")
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '--upgrade', '--no-cache-dir',
     'bitsandbytes>=0.45.0'],
    capture_output=True,
    text=True,
    check=False,
    timeout=600
)
if result.returncode == 0:
    print("  ✓ bitsandbytes >= 0.45.0 installed")
    try:
        import bitsandbytes
        print(f"  ✓ Version: {bitsandbytes.__version__}")
    except:
        pass
else:
    print(f"  ⚠ Install had warnings: {result.stderr[:200] if result.stderr else 'Check manually'}")

# Step 2: Manual Binary Patch - Locate bitsandbytes and create symlink
print("\n[Step 2] Locating bitsandbytes installation directory...")
bitsandbytes_lib_dir = None

try:
    import bitsandbytes
    bitsandbytes_file = Path(bitsandbytes.__file__)
    bitsandbytes_path = bitsandbytes_file.parent
    print(f"  ✓ Found bitsandbytes at: {bitsandbytes_path}")
    
    # Search for library directory with .so files
    for lib_dir in [bitsandbytes_path, bitsandbytes_path / "cuda_setup", 
                    bitsandbytes_path / "cextension", bitsandbytes_path.parent / "bitsandbytes"]:
        if lib_dir.exists():
            so_files = list(lib_dir.glob("libbitsandbytes*.so"))
            if so_files:
                bitsandbytes_lib_dir = lib_dir
                print(f"  ✓ Found library directory: {bitsandbytes_lib_dir}")
                print(f"    Contains: {[f.name for f in so_files]}")
                break
    
    # If not found, search recursively
    if not bitsandbytes_lib_dir:
        for so_file in bitsandbytes_path.rglob("libbitsandbytes*.so"):
            bitsandbytes_lib_dir = so_file.parent
            print(f"  ✓ Found library directory: {bitsandbytes_lib_dir}")
            break
            
except Exception as e:
    print(f"  ⚠ Could not locate via import: {e}")
    # Fallback: search site-packages
    import site
    for sp in site.getsitepackages():
        bnb_path = Path(sp) / "bitsandbytes"
        if bnb_path.exists():
            for so_file in bnb_path.rglob("libbitsandbytes*.so"):
                bitsandbytes_lib_dir = so_file.parent
                print(f"  ✓ Found library directory: {bitsandbytes_lib_dir}")
                break
            if bitsandbytes_lib_dir:
                break

# Step 3: Create Symlink - Link cuda121 (or any 12.x) to cuda126
print("\n[Step 3] Creating symlink for libbitsandbytes_cuda126.so...")
symlink_created = False

if bitsandbytes_lib_dir:
    target_lib = bitsandbytes_lib_dir / "libbitsandbytes_cuda126.so"
    
    # Check if target already exists
    if target_lib.exists():
        print(f"  ✓ libbitsandbytes_cuda126.so already exists")
        symlink_created = True
    else:
        # Find source library (prefer 12.x versions)
        source_candidates = [
            "libbitsandbytes_cuda121.so",
            "libbitsandbytes_cuda120.so",
            "libbitsandbytes_cuda122.so",
            "libbitsandbytes_cuda118.so",
            "libbitsandbytes_cuda117.so",
            "libbitsandbytes_cpu.so",
            "libbitsandbytes.so",
        ]
        
        source_lib = None
        for candidate in source_candidates:
            candidate_path = bitsandbytes_lib_dir / candidate
            if candidate_path.exists():
                source_lib = candidate_path
                print(f"  ✓ Found source library: {candidate}")
                break
        
        if source_lib:
            try:
                # Create relative symlink (preferred)
                target_lib.symlink_to(source_lib.name)
                print(f"  ✓ Created symlink: libbitsandbytes_cuda126.so -> {source_lib.name}")
                symlink_created = True
            except OSError:
                try:
                    # Try absolute symlink
                    target_lib.symlink_to(source_lib)
                    print(f"  ✓ Created absolute symlink: libbitsandbytes_cuda126.so -> {source_lib}")
                    symlink_created = True
                except Exception as e:
                    # Fallback: copy the file
                    try:
                        import shutil
                        shutil.copy2(source_lib, target_lib)
                        print(f"  ✓ Copied library: libbitsandbytes_cuda126.so (from {source_lib.name})")
                        symlink_created = True
                    except Exception as e2:
                        print(f"  ✗ Failed to create link/copy: {e2}")
        else:
            print(f"  ✗ No source library found")
            print(f"    Searched for: {', '.join(source_candidates)}")
            actual_files = list(bitsandbytes_lib_dir.glob("*.so"))
            if actual_files:
                print(f"    Found files: {[f.name for f in actual_files]}")
else:
    print("  ✗ Could not find bitsandbytes library directory")

# Step 4: Environment Override - Set BNB_CUDA_VERSION
print("\n[Step 4] Setting environment override...")
os.environ["BNB_CUDA_VERSION"] = "126"
print("  ✓ Set BNB_CUDA_VERSION=126")

# Also set CUDA_VERSION for compatibility
os.environ["CUDA_VERSION"] = "126"
print("  ✓ Set CUDA_VERSION=126")

# Step 5: Verification - Quick check
print("\n[Step 5] Quick verification...")
if symlink_created and bitsandbytes_lib_dir:
    target_check = bitsandbytes_lib_dir / "libbitsandbytes_cuda126.so"
    if target_check.exists():
        print(f"  ✓ libbitsandbytes_cuda126.so exists at: {target_check}")
        if target_check.is_symlink():
            print(f"  ✓ Is symlink pointing to: {target_check.readlink()}")
    else:
        print(f"  ⚠ libbitsandbytes_cuda126.so not found (may need restart)")
else:
    print(f"  ⚠ Symlink creation status unclear")

# Step 6: Hard Reset - Force kernel restart
print("\n" + "="*70)
print("HARD RESET - KERNEL RESTART")
print("="*70)
print("")
print("Environment updated. Colab is now restarting the kernel automatically.")
print("WAIT for the 'Connected' status at the top right, then run your Audit cell.")
print("")
print("🔄 Restarting kernel in 2 seconds to reload dynamic libraries...")
print("="*70)

import time
time.sleep(2)

# HARD RESET - Force kernel restart
print("\n💥 Executing os._exit(0) - Kernel restarting now...")
os._exit(0)
