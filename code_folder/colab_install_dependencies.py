# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================
# Install all necessary libraries for Llama-3 8B with A100 optimization
# ============================================================================

!pip install -q -U transformers accelerate bitsandbytes flash-attn --no-build-isolation

print("="*70)
print("✓ Dependencies installed successfully")
print("="*70)
