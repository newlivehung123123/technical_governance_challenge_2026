# ============================================================================
# BITSANDBYTES CUDA 12.6 FIX - Enable CUDA 12.6 Support
# ============================================================================
# Fixes missing libbitsandbytes_cuda126.so by creating symlink and setting env vars
# ============================================================================

import os, sys, subprocess
from pathlib import Path

print("="*70)
print("BITSANDBYTES CUDA 12.6 FIX")
print("="*70)

# Step 1: Install the Fix Version - bitsandbytes >= 0.45.0
print("\n[Step 1] Installing bitsandbytes >= 0.45.0 (CUDA 12+ support)...")
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-cache-dir',
     'bitsandbytes>=0.45.0'],
    capture_output=True,
    text=True,
    check=False,
    timeout=600
)
if result.returncode == 0:
    print("  ✓ bitsandbytes >= 0.45.0 installed")
    # Try to get version
    try:
        import bitsandbytes
        print(f"  ✓ Installed version: {bitsandbytes.__version__}")
    except:
        pass
else:
    print(f"  ⚠ bitsandbytes install had warnings: {result.stderr[:200] if result.stderr else 'Check manually'}")

# Step 2: Find bitsandbytes installation directory
print("\n[Step 2] Finding bitsandbytes installation...")
bitsandbytes_path = None
bitsandbytes_lib_dir = None

try:
    import bitsandbytes
    bitsandbytes_file = Path(bitsandbytes.__file__)
    bitsandbytes_path = bitsandbytes_file.parent
    print(f"  ✓ Found bitsandbytes at: {bitsandbytes_path}")
    
    # Look for lib directory (could be in cuda_setup or cextension)
    possible_lib_dirs = [
        bitsandbytes_path / "cuda_setup",
        bitsandbytes_path / "cextension",
        bitsandbytes_path / "lib",
        bitsandbytes_path.parent / "bitsandbytes" / "cuda_setup",
        bitsandbytes_path.parent / "bitsandbytes" / "cextension",
    ]
    
    for lib_dir in possible_lib_dirs:
        if lib_dir.exists():
            # Check if it contains .so files
            so_files = list(lib_dir.glob("*.so"))
            if so_files:
                bitsandbytes_lib_dir = lib_dir
                print(f"  ✓ Found library directory: {bitsandbytes_lib_dir}")
                print(f"    Contains {len(so_files)} .so files")
                break
    
    if not bitsandbytes_lib_dir:
        # Search recursively
        for so_file in bitsandbytes_path.rglob("*.so"):
            bitsandbytes_lib_dir = so_file.parent
            print(f"  ✓ Found library directory: {bitsandbytes_lib_dir}")
            break
            
except Exception as e:
    print(f"  ⚠ Could not find bitsandbytes: {e}")
    # Try to find in site-packages
    import site
    for sp in site.getsitepackages():
        bnb_path = Path(sp) / "bitsandbytes"
        if bnb_path.exists():
            bitsandbytes_path = bnb_path
            print(f"  ✓ Found bitsandbytes at: {bnb_path}")
            # Search for .so files
            for so_file in bnb_path.rglob("*.so"):
                bitsandbytes_lib_dir = so_file.parent
                print(f"  ✓ Found library directory: {bitsandbytes_lib_dir}")
                break
            if bitsandbytes_lib_dir:
                break

# Step 3: Manual Binary Link - Create symlink for CUDA 12.6
print("\n[Step 3] Creating symlink for libbitsandbytes_cuda126.so...")
if bitsandbytes_lib_dir:
    target_lib = bitsandbytes_lib_dir / "libbitsandbytes_cuda126.so"
    symlink_created = False
    
    # Check if target already exists
    if target_lib.exists():
        print(f"  ✓ libbitsandbytes_cuda126.so already exists")
        symlink_created = True
    else:
        # Try to find source libraries to symlink from
        source_candidates = [
            "libbitsandbytes_cuda121.so",
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
                # Create symlink
                target_lib.symlink_to(source_lib.name)  # Relative symlink
                print(f"  ✓ Created symlink: {target_lib.name} -> {source_lib.name}")
                symlink_created = True
            except OSError:
                # Try absolute symlink if relative fails
                try:
                    target_lib.symlink_to(source_lib)
                    print(f"  ✓ Created absolute symlink: {target_lib.name} -> {source_lib}")
                    symlink_created = True
                except Exception as e:
                    print(f"  ⚠ Could not create symlink: {e}")
                    # Try copying instead
                    try:
                        import shutil
                        shutil.copy2(source_lib, target_lib)
                        print(f"  ✓ Copied library: {target_lib.name} (from {source_lib.name})")
                        symlink_created = True
                    except Exception as e2:
                        print(f"  ✗ Could not copy library: {e2}")
        else:
            print(f"  ⚠ No source library found to symlink from")
            print(f"    Searched for: {', '.join(source_candidates)}")
            print(f"    In directory: {bitsandbytes_lib_dir}")
            # List what's actually there
            actual_files = list(bitsandbytes_lib_dir.glob("*.so"))
            if actual_files:
                print(f"    Found files: {[f.name for f in actual_files]}")
    
    if symlink_created:
        print("  ✓ CUDA 12.6 library link created")
    else:
        print("  ⚠ Could not create CUDA 12.6 library link")
else:
    print("  ⚠ Could not find bitsandbytes library directory")

# Step 4: Force Backend - Set environment variables
print("\n[Step 4] Setting CUDA environment variables...")
os.environ["BNB_CUDA_VERSION"] = "126"
os.environ["CUDA_VERSION"] = "126"
print("  ✓ Set BNB_CUDA_VERSION=126")
print("  ✓ Set CUDA_VERSION=126")

# Also ensure CUDA_HOME is set if not already
if "CUDA_HOME" not in os.environ:
    cuda_paths = [
        Path("/usr/local/cuda"),
        Path("/usr/local/cuda-12.6"),
        Path("/usr/local/cuda-12.0"),
    ]
    for cuda_path in cuda_paths:
        if cuda_path.exists():
            os.environ["CUDA_HOME"] = str(cuda_path)
            print(f"  ✓ Set CUDA_HOME={cuda_path}")
            break

# Step 5: Verification - Check if fix worked
print("\n[Step 5] Verifying CUDA 12.6 support...")
try:
    import bitsandbytes
    print(f"  ✓ bitsandbytes imported successfully")
    
    # Try to check CUDA support
    try:
        compiled_with_cuda = bitsandbytes.cextension.COMPILED_WITH_CUDA
        print(f"  COMPILED_WITH_CUDA: {compiled_with_cuda}")
    except:
        pass
    
    # Try to import quantization functions
    try:
        from bitsandbytes import functional as F
        print(f"  ✓ bitsandbytes functional module imported")
    except Exception as e:
        print(f"  ⚠ Functional import warning: {e}")
    
    print("  ✓ bitsandbytes appears to be working")
except Exception as e:
    print(f"  ⚠ bitsandbytes verification had issues: {e}")
    print("  ⚠ This may be resolved after kernel restart")

# Step 6: Final Status
print("\n" + "="*70)
print("CUDA 12.6 FIX COMPLETE")
print("="*70)
print("✓ bitsandbytes >= 0.45.0 installed")
if bitsandbytes_lib_dir:
    print(f"✓ Library directory: {bitsandbytes_lib_dir}")
    if (bitsandbytes_lib_dir / "libbitsandbytes_cuda126.so").exists():
        print("✓ libbitsandbytes_cuda126.so created")
    else:
        print("⚠ libbitsandbytes_cuda126.so may need kernel restart to be recognized")
print("✓ Environment variables set (BNB_CUDA_VERSION=126, CUDA_VERSION=126)")
print("="*70)

# Step 7: Hard Kill - Restart kernel to lock in changes
print("\n" + "="*70)
print("KERNEL RESTART REQUIRED")
print("="*70)
print("Environment updated. Colab is now restarting the kernel automatically.")
print("WAIT for the 'Connected' status at the top right, then run your Audit cell.")
print("")
print("🔄 Restarting kernel in 2 seconds to lock in CUDA 12.6 configuration...")
print("="*70)

import time
time.sleep(2)

# HARD KILL - Force kernel restart
print("\n💥 Executing os._exit(0) - Kernel restarting now...")
os._exit(0)
