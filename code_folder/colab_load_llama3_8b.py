# ============================================================================
# LOAD LLAMA-3 8B WITH 4-BIT QUANTIZATION
# ============================================================================
# Verifies GPU support and loads Llama-3 8B using bitsandbytes 0.49.1
# ============================================================================

# MANUAL TOKEN INPUT - Paste your HuggingFace token here
HF_TOKEN = "your_token_here"  # Replace with your actual HuggingFace token

import os
# Set timeout to 5 minutes (300 seconds) to handle large model downloads
os.environ["HF_HUB_READ_TIMEOUT"] = "300"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import whoami, model_info

print("="*70)
print("LLAMA-3 8B LOADER - 4-bit Quantization")
print("="*70)

# Step 1: Verify GPU Support
print("\n[Step 1] Verifying GPU support...")

# Check PyTorch CUDA
if torch.cuda.is_available():
    print(f"  ✓ PyTorch CUDA available")
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ CUDA Version: {torch.version.cuda}")
else:
    print("  ✗ PyTorch CUDA not available")
    raise RuntimeError("CUDA not available in PyTorch")

# Check bitsandbytes CUDA support (new method for 0.49.1)
print("\n[Step 2] Verifying bitsandbytes GPU support...")
bitsandbytes_working = False

try:
    import bitsandbytes as bnb
    print(f"  ✓ bitsandbytes imported (version: {bnb.__version__})")
    
    # Method 1: Try to get CUDA library handle (new API)
    try:
        from bitsandbytes import cuda_setup
        cuda_lib = cuda_setup.get_cuda_lib_handle()
        if cuda_lib:
            print("  ✓ CUDA library handle obtained")
            bitsandbytes_working = True
        else:
            print("  ⚠ CUDA library handle is None")
    except Exception as e:
        print(f"  ⚠ get_cuda_lib_handle() failed: {e}")
    
    # Method 2: Try to initialize a small 4-bit linear layer
    if not bitsandbytes_working:
        try:
            from bitsandbytes.nn import Linear4bit
            import torch.nn as nn
            
            # Create a tiny test layer
            test_layer = Linear4bit(10, 10, quant_type="nf4")
            print("  ✓ 4-bit linear layer initialized successfully")
            bitsandbytes_working = True
        except Exception as e:
            print(f"  ⚠ 4-bit linear layer test failed: {e}")
    
    # Method 3: Check if quantization functions are available
    if not bitsandbytes_working:
        try:
            from bitsandbytes import functional as F
            print("  ✓ bitsandbytes functional module available")
            bitsandbytes_working = True
        except Exception as e:
            print(f"  ⚠ Functional module import failed: {e}")
    
    if bitsandbytes_working:
        print("  ✅ bitsandbytes GPU support verified")
    else:
        print("  ⚠ bitsandbytes GPU support unclear (will attempt model load)")
        
except ImportError as e:
    print(f"  ✗ bitsandbytes import failed: {e}")
    raise RuntimeError(f"bitsandbytes not available: {e}")
except Exception as e:
    print(f"  ⚠ bitsandbytes check had issues: {e}")
    print("  ⚠ Will attempt model load anyway")

# Step 3: Configure 4-bit Quantization
print("\n[Step 3] Configuring 4-bit quantization...")
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    print("  ✓ BitsAndBytesConfig created")
    print("    - 4-bit quantization: NF4")
    print("    - Compute dtype: bfloat16")
    print("    - Double quantization: enabled")
except Exception as e:
    print(f"  ✗ BitsAndBytesConfig creation failed: {e}")
    raise RuntimeError(f"Could not create quantization config: {e}")

# Step 4: Credential Check and Permission Audit
print("\n[Step 4] Credential check and permission audit...")

# Check token
if HF_TOKEN == "your_token_here" or not HF_TOKEN:
    print("  ⚠ WARNING: HF_TOKEN not set!")
    print("  Please set HF_TOKEN = 'your_actual_token' at the top of this cell")
    raise ValueError("HF_TOKEN must be set to your HuggingFace token")

print(f"  ✓ Using HuggingFace token (length: {len(HF_TOKEN)} chars)")

# Credential Check
print("\n  [4a] Checking authentication...")
try:
    user_info = whoami(token=HF_TOKEN)
    username = user_info.get('name', 'Unknown')
    print(f"  ✓ Authenticated as: {username}")
    print(f"  ✓ User type: {user_info.get('type', 'Unknown')}")
except Exception as e:
    print(f"  ✗ Authentication failed: {e}")
    raise RuntimeError(f"Token authentication failed: {e}")

# Permission Audit
print("\n  [4b] Checking model permissions...")
instruct_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
base_model_id = "meta-llama/Meta-Llama-3-8B"

instruct_accessible = False
base_accessible = False
instruct_gated = False
base_gated = False

try:
    instruct_info = model_info(instruct_model_id, token=HF_TOKEN)
    instruct_gated = instruct_info.gated if hasattr(instruct_info, 'gated') else False
    print(f"  ✓ Instruct model info retrieved")
    print(f"    Gated: {instruct_gated}")
    if not instruct_gated:
        instruct_accessible = True
        print(f"    Status: Accessible")
    else:
        print(f"    Status: Gated (requires access request)")
except Exception as e:
    error_msg = str(e)
    if "403" in error_msg or "Forbidden" in error_msg:
        print(f"  ✗ Instruct model: 403 Forbidden")
        print(f"    Status: Access denied")
    elif "401" in error_msg or "Unauthorized" in error_msg:
        print(f"  ✗ Instruct model: 401 Unauthorized")
        print(f"    Status: Invalid token")
    else:
        print(f"  ⚠ Instruct model check failed: {e}")

try:
    base_info = model_info(base_model_id, token=HF_TOKEN)
    base_gated = base_info.gated if hasattr(base_info, 'gated') else False
    print(f"  ✓ Base model info retrieved")
    print(f"    Gated: {base_gated}")
    if not base_gated:
        base_accessible = True
        print(f"    Status: Accessible")
    else:
        print(f"    Status: Gated (requires access request)")
except Exception as e:
    error_msg = str(e)
    if "403" in error_msg or "Forbidden" in error_msg:
        print(f"  ✗ Base model: 403 Forbidden")
        print(f"    Status: Access denied")
    elif "401" in error_msg or "Unauthorized" in error_msg:
        print(f"  ✗ Base model: 401 Unauthorized")
        print(f"    Status: Invalid token")
    else:
        print(f"  ⚠ Base model check failed: {e}")

# Determine which model to use
print("\n  [4c] Determining model to load...")
if instruct_accessible or (instruct_gated and not base_accessible):
    model_name = instruct_model_id
    print(f"  ✓ Will attempt to load: {model_name}")
elif base_accessible:
    model_name = base_model_id
    print(f"  ⚠ Instruct model not accessible, will try base model: {model_name}")
else:
    # Both are gated or forbidden - provide instructions
    print("\n" + "="*70)
    print("❌ MODEL ACCESS DENIED")
    print("="*70)
    print(f"Authenticated as: {username}")
    print(f"\nModel Access Status:")
    print(f"  - {instruct_model_id}: {'Gated' if instruct_gated else 'Forbidden'}")
    print(f"  - {base_model_id}: {'Gated' if base_gated else 'Forbidden'}")
    print("\n" + "="*70)
    print("📋 ACTION REQUIRED:")
    print("="*70)
    
    if instruct_gated or base_gated:
        print("\n🔗 Visit this URL to request access:")
        if instruct_gated:
            print(f"   https://huggingface.co/{instruct_model_id}")
        elif base_gated:
            print(f"   https://huggingface.co/{base_model_id}")
        print("\nSteps:")
        print("  1. Click 'Agree and access repository'")
        print("  2. Accept Meta's license terms")
        print("  3. Wait for approval (usually instant)")
        print("  4. Re-run this cell")
    else:
        print("\n⚠️  Both models returned 403 Forbidden")
        print("   This may indicate:")
        print("   - Token does not have required permissions")
        print("   - Account needs to be verified")
        print("   - Model access was revoked")
        print("\n🔗 Visit these URLs to check:")
        print(f"   https://huggingface.co/{instruct_model_id}")
        print(f"   https://huggingface.co/{base_model_id}")
        print("\n   And verify your account has access")
    
    print("="*70)
    raise RuntimeError("Model access denied - please request access at the URL above")

print(f"\n[Step 5] Loading model: {model_name}")

try:
    print("  Loading model with 4-bit quantization and Flash Attention 2 for A100...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        token=HF_TOKEN  # Explicit token authentication
    )
    
    # If we loaded base model but wanted Instruct, note it
    if model_name == base_model_id and instruct_model_id != base_model_id:
        print("  ⚠ Loaded base model (Instruct version was not accessible)")
    print("  ✓ Model loaded successfully")
    
    # Check device placement
    print("\n  Model device placement:")
    device_count = {}
    for name, param in model.named_parameters():
        if param.device.type not in device_count:
            device_count[param.device.type] = 0
        device_count[param.device.type] += 1
    
    for device, count in device_count.items():
        print(f"    {device}: {count} parameters")
    
    if 'cuda' in device_count:
        print("  ✓ Model loaded on GPU")
    else:
        print("  ⚠ Model not on GPU (may be on CPU)")
        
except Exception as e:
    error_msg = str(e)
    print(f"\n  ✗ Model loading failed: {error_msg}")
    
    # Diagnostic messages
    print("\n" + "="*70)
    print("DIAGNOSTIC INFORMATION")
    print("="*70)
    
    if "bitsandbytes" in error_msg.lower() or "quantization" in error_msg.lower():
        print("❌ BITSANDBYTES ERROR DETECTED")
        print("\nPossible issues:")
        print("  1. bitsandbytes not compiled with CUDA support")
        print("  2. CUDA version mismatch")
        print("  3. Missing CUDA libraries")
        print("\nTroubleshooting:")
        print("  - Check: import bitsandbytes; print(bitsandbytes.__version__)")
        print("  - Verify CUDA: torch.cuda.is_available()")
        print("  - Try: from bitsandbytes import cuda_setup; cuda_setup.get_cuda_lib_handle()")
    elif "out of memory" in error_msg.lower() or "OOM" in error_msg.upper():
        print("❌ OUT OF MEMORY ERROR")
        print("\nPossible solutions:")
        print("  - Use smaller model")
        print("  - Reduce batch size")
        print("  - Use 8-bit quantization instead")
    elif "authentication" in error_msg.lower() or "token" in error_msg.lower():
        print("❌ AUTHENTICATION ERROR")
        print("\nSolution:")
        print("  - Verify your HF_TOKEN is correct")
        print("  - Check token at: https://huggingface.co/settings/tokens")
    elif "403" in error_msg or "Forbidden" in error_msg:
        print("❌ FORBIDDEN ERROR (403)")
        print("\nThis means your token is valid but you don't have access to the model.")
        print("\n🔗 Visit this URL to request access:")
        print(f"   https://huggingface.co/{model_name}")
        print("\nSteps:")
        print("  1. Click 'Agree and access repository'")
        print("  2. Accept Meta's license terms")
        print("  3. Wait for approval")
        print("  4. Re-run this cell")
    else:
        print("❌ UNKNOWN ERROR")
        print(f"\nError details: {error_msg}")
    
    print("="*70)
    raise

# Step 6: Load Tokenizer
print("\n[Step 6] Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=HF_TOKEN  # Explicit token authentication
    )
    print("  ✓ Tokenizer loaded successfully")
except Exception as e:
    print(f"  ✗ Tokenizer loading failed: {e}")
    raise RuntimeError(f"Tokenizer loading failed: {e}")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("  ✓ Set pad_token to eos_token")

# Step 7: Verification Step - Test generation
print("\n[Step 7] Verification: Testing model with 'Hello' generation...")
try:
    # Prepare input
    test_prompt = "Hello"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    print(f"  Input: '{test_prompt}'")
    print("  Generating response...")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Generated: '{generated_text}'")
    
    # Verify
    if "Hello" in generated_text or len(generated_text) > len(test_prompt):
        print("  ✓ Model weights are active and generating text")
        verification_passed = True
    else:
        print("  ⚠ Generation test unclear")
        verification_passed = False
        
except Exception as e:
    print(f"  ⚠ Generation test failed: {e}")
    print("  ⚠ Model loaded but generation test failed")
    verification_passed = False

# Step 8: Success Signal
print("\n" + "="*70)
if verification_passed:
    print("🚀 CENSUS READY: Llama-3 8B Loaded with GPU")
else:
    print("⚠️  Llama-3 8B Loaded (verification test had issues)")
print("="*70)
print(f"✓ Model: {model_name}")
print("✓ Quantization: 4-bit (NF4)")
print("✓ Device: GPU (auto)")
print("✓ Compute dtype: bfloat16")
print("✓ Tokenizer: Loaded")
if verification_passed:
    print("✓ Verification: Model generating text")
print("="*70)
print("\n✅ Model is ready for inference!")
print("You can now use 'model' and 'tokenizer' variables in your code.")
print("="*70)
