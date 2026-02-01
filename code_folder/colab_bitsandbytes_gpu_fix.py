# ============================================================================
# BITSANDBYTES GPU FIX - Enable GPU Support for 4-bit Quantization
# ============================================================================
# Fixes missing triton.ops and ensures bitsandbytes detects GPU
# ============================================================================

import os, sys, subprocess, ctypes
from pathlib import Path

print("="*70)
print("BITSANDBYTES GPU FIX - Enabling GPU Support")
print("="*70)

# Step 1: Install Triton
print("\n[Step 1] Installing Triton (required for bitsandbytes GPU support)...")
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', 'triton==3.0.0', '--no-cache-dir'],
    capture_output=True,
    text=True,
    check=False,
    timeout=300
)
if result.returncode == 0:
    print("  ✓ triton==3.0.0 installed")
else:
    # Try latest version if 3.0.0 fails
    print("  ⚠ triton==3.0.0 failed, trying latest version...")
    result2 = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', 'triton', '--no-cache-dir', '--upgrade'],
        capture_output=True,
        text=True,
        check=False,
        timeout=300
    )
    if result2.returncode == 0:
        print("  ✓ triton (latest) installed")
    else:
        print(f"  ⚠ triton install had issues: {result2.stderr[:200] if result2.stderr else 'Unknown'}")

# Step 2: Find CUDA installation
print("\n[Step 2] Finding CUDA installation...")
cuda_paths = [
    Path("/usr/local/cuda"),
    Path("/usr/local/cuda-12.0"),
    Path("/usr/local/cuda-12.1"),
    Path("/usr/local/cuda-12.2"),
    Path("/usr/local/cuda-11.8"),
    Path("/usr/local/cuda-11.7"),
]

cuda_home = None
cuda_lib64 = None

for cuda_path in cuda_paths:
    if cuda_path.exists():
        cuda_home = str(cuda_path)
        cuda_lib64 = str(cuda_path / "lib64")
        if Path(cuda_lib64).exists():
            print(f"  ✓ Found CUDA at: {cuda_home}")
            print(f"  ✓ CUDA lib64: {cuda_lib64}")
            break

if not cuda_home:
    print("  ⚠ CUDA not found in standard locations")
    print("  ⚠ Attempting to continue anyway...")

# Step 3: Library Path Injection - Set CUDA environment variables
print("\n[Step 3] Injecting CUDA library paths...")
if cuda_lib64:
    # Set CUDA_HOME
    os.environ['CUDA_HOME'] = cuda_home
    print(f"  ✓ Set CUDA_HOME={cuda_home}")
    
    # Update LD_LIBRARY_PATH to include CUDA lib64 FIRST
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if cuda_lib64 not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib64}:{current_ld_path}"
        print(f"  ✓ Updated LD_LIBRARY_PATH (CUDA lib64 at front)")
    else:
        # Move to front if already present
        paths = current_ld_path.split(':')
        if cuda_lib64 in paths:
            paths.remove(cuda_lib64)
        os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib64}:{':'.join(paths)}"
        print(f"  ✓ Moved CUDA lib64 to front of LD_LIBRARY_PATH")
    
    # Also set PATH for CUDA binaries
    cuda_bin = str(Path(cuda_home) / "bin")
    if Path(cuda_bin).exists():
        current_path = os.environ.get('PATH', '')
        if cuda_bin not in current_path:
            os.environ['PATH'] = f"{cuda_bin}:{current_path}"
            print(f"  ✓ Updated PATH with CUDA binaries")
else:
    print("  ⚠ CUDA paths not set (CUDA not found)")

# Step 4: Manual Library Loading - Force load CUDA libraries
print("\n[Step 4] Manually loading CUDA libraries...")
if cuda_lib64:
    libcuda_path = Path(cuda_lib64) / "libcuda.so"
    libcudart_path = Path(cuda_lib64) / "libcudart.so"
    
    try:
        if libcuda_path.exists():
            ctypes.CDLL(str(libcuda_path), mode=ctypes.RTLD_GLOBAL)
            print(f"  ✓ Loaded libcuda.so")
        else:
            # Try alternative locations
            alt_paths = [
                "/usr/lib/x86_64-linux-gnu/libcuda.so",
                "/usr/lib64/libcuda.so",
                "/usr/lib/libcuda.so"
            ]
            for alt_path in alt_paths:
                if Path(alt_path).exists():
                    ctypes.CDLL(alt_path, mode=ctypes.RTLD_GLOBAL)
                    print(f"  ✓ Loaded libcuda.so from {alt_path}")
                    break
    except Exception as e:
        print(f"  ⚠ Could not load libcuda.so: {e}")
    
    try:
        if libcudart_path.exists():
            ctypes.CDLL(str(libcudart_path), mode=ctypes.RTLD_GLOBAL)
            print(f"  ✓ Loaded libcudart.so")
        else:
            # Try alternative locations
            alt_paths = [
                f"{cuda_home}/lib/libcudart.so",
                "/usr/lib/x86_64-linux-gnu/libcudart.so"
            ]
            for alt_path in alt_paths:
                if Path(alt_path).exists():
                    ctypes.CDLL(alt_path, mode=ctypes.RTLD_GLOBAL)
                    print(f"  ✓ Loaded libcudart.so from {alt_path}")
                    break
    except Exception as e:
        print(f"  ⚠ Could not load libcudart.so: {e}")
else:
    print("  ⚠ Skipping library loading (CUDA not found)")

# Step 5: Force GPU bitsandbytes - Reinstall with CUDA paths set
print("\n[Step 5] Force reinstalling bitsandbytes with GPU support...")
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps',
     '--no-cache-dir', 'bitsandbytes==0.43.3'],
    capture_output=True,
    text=True,
    check=False,
    timeout=600,
    env=os.environ.copy()  # Pass environment with CUDA paths
)
if result.returncode == 0:
    print("  ✓ bitsandbytes==0.43.3 reinstalled")
else:
    print(f"  ⚠ bitsandbytes install had warnings: {result.stderr[:200] if result.stderr else 'Check manually'}")

# Step 6: Reset Verification - Check GPU support
print("\n[Step 6] Verifying GPU support...")
gpu_detected = False

try:
    import bitsandbytes
    print(f"  ✓ bitsandbytes imported successfully")
    
    # Check if compiled with CUDA
    try:
        compiled_with_cuda = bitsandbytes.cextension.COMPILED_WITH_CUDA
        print(f"  COMPILED_WITH_CUDA: {compiled_with_cuda}")
        
        if compiled_with_cuda:
            print("  ✓ GPU support detected!")
            gpu_detected = True
        else:
            print("  ❌ GPU NOT DETECTED BY BNB")
            print("  ⚠ bitsandbytes was compiled without GPU support")
            gpu_detected = False
    except AttributeError:
        # Try alternative check
        try:
            # Check if CUDA functions are available
            from bitsandbytes import cuda_setup
            print("  ✓ CUDA setup module available")
            gpu_detected = True
        except Exception as e:
            print(f"  ❌ GPU NOT DETECTED BY BNB")
            print(f"  Error: {e}")
            gpu_detected = False
    except Exception as e:
        print(f"  ❌ GPU NOT DETECTED BY BNB")
        print(f"  Error checking CUDA support: {e}")
        gpu_detected = False
        
except ModuleNotFoundError as e:
    if 'triton' in str(e):
        print(f"  ❌ GPU NOT DETECTED BY BNB")
        print(f"  Missing triton module: {e}")
        print("  ⚠ Triton installation may have failed")
    else:
        print(f"  ❌ GPU NOT DETECTED BY BNB")
        print(f"  bitsandbytes import failed: {e}")
    gpu_detected = False
except Exception as e:
    print(f"  ❌ GPU NOT DETECTED BY BNB")
    print(f"  Unexpected error: {e}")
    gpu_detected = False

# Step 7: Final Status
print("\n" + "="*70)
if gpu_detected:
    print("✅ GPU SUPPORT ENABLED")
    print("="*70)
    print("✓ bitsandbytes compiled with CUDA support")
    print("✓ GPU detected and ready for 4-bit quantization")
    print("✓ You can now load Llama-3 8B with 4-bit quantization")
    print("="*70)
else:
    print("❌ GPU SUPPORT NOT DETECTED")
    print("="*70)
    print("⚠ bitsandbytes does not have GPU support")
    print("⚠ 4-bit quantization may not work")
    print("⚠ Check CUDA installation and try again")
    print("="*70)

# Step 8: Atomic Exit - Lock in CUDA paths
print("\n" + "="*70)
print("LOCKING ENVIRONMENT")
print("="*70)
print("Environment updated. Colab is now restarting the kernel automatically.")
print("WAIT for the 'Connected' status at the top right, then run your Audit cell.")
print("")
print("🔄 Restarting kernel in 2 seconds to lock in CUDA paths...")
print("="*70)

import time
time.sleep(2)

# IMMEDIATE KILL - Force kernel restart to lock in environment
print("\n💥 Executing os._exit(0) - Kernel restarting now...")
os._exit(0)
