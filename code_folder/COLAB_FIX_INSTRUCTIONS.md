# Google Colab Dependency Fix - Instructions

## Quick Start

1. **Open a new cell in Google Colab**
2. **Copy the entire contents of `colab_fix_single_cell.py`**
3. **Paste into the cell and run it**
4. **Wait for completion** (takes ~2-3 minutes)
5. **In the next cell, test imports:**
   ```python
   import numpy as np
   import pandas as pd
   import transformers
   print("All imports successful!")
   ```

## What This Fix Does

1. **Disables debugger crash loops** - Sets `PYDEVD_DISABLE_FILE_VALIDATION=1`
2. **Purges ghost folders** - Removes corrupted `~` folders in dist-packages
3. **Clean uninstall** - Removes conflicting numpy/pandas/transformers versions
4. **Surgical install** - Installs exact versions with `--no-deps`:
   - numpy==2.1.0 (NumPy 2.x native, fixes dtype size mismatch)
   - pandas==2.2.2 (Colab requirement)
   - transformers==4.44.2 (compatible with bitsandbytes)
   - tokenizers==0.20.0 (compatible version)
   - bitsandbytes==0.43.3 (4-bit quantization support)
5. **Refreshes Python path** - Uses `site.addsitedir` to reload without kernel restart

## Troubleshooting

### If imports still fail:
1. **Restart runtime ONCE** (Runtime → Restart runtime)
2. **Re-run the fix cell**
3. **Test imports again**

### If kernel keeps crashing:
- The fix cell should prevent this, but if it persists:
  1. Go to Runtime → Change runtime type
  2. Ensure GPU is selected (if using bitsandbytes)
  3. Restart runtime
  4. Run fix cell again

### If bitsandbytes fails:
- This is normal if CUDA is not available
- The other packages (numpy, pandas, transformers) should still work
- For 4-bit quantization, you need a GPU runtime

## Expected Output

You should see:
```
✓ Debugger validation disabled
✓ Purged X ghost folders...
✓ Cleaned conflicting packages
  ✓ numpy==2.1.0
  ✓ pandas==2.2.2
  ✓ transformers==4.44.2
  ...
✓ Dependency fix complete! You can now import in the next cell.
```

## Notes

- **Don't restart kernel multiple times** - This can cause loops
- **Run the fix cell only once** - Re-running is safe but unnecessary
- **The fix persists** - Once fixed, imports work in subsequent cells
- **GPU recommended** - For bitsandbytes 4-bit quantization, use GPU runtime
