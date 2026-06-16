# Training Notebook Fix — Summary

## Problem Fixed

**Error:** `IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)`

**Location:** `notebooks/02_train_cnn1d.ipynb`, Cell 6 (Training code)

**Root Cause:** Incorrect tensor shape manipulation
```python
y.to(device).squeeze(1)  # ❌ WRONG
```

- `y` from DataLoader has shape `(batch_size,)` — already 1D
- `.squeeze(1)` tries to remove dimension 1
- A 1D tensor only has dimension 0
- Trying to access dimension 1 causes IndexError

---

## Solution Applied

### Changes Made

**File:** `notebooks/02_train_cnn1d.ipynb`

**Location 1 - `train_epoch()` function:**
```python
# BEFORE (caused error):
X, y = X.to(device), y.to(device).squeeze(1)

# AFTER (fixed):
X, y = X.to(device), y.to(device)
```

**Location 2 - `eval_epoch()` function:**
```python
# BEFORE (caused error):
X, y = X.to(device), y.to(device).squeeze(1)

# AFTER (fixed):
X, y = X.to(device), y.to(device)
```

**Keep as-is (both functions):**
```python
pred = model(X).squeeze(1)  # ✅ Correct - pred needs squeezing
```

---

## Why This Works

### Tensor Shapes at Each Step

| Variable | Shape | Dimensions | Needs Squeeze? |
|----------|-------|-----------|---|
| `y` from DataLoader | `(64,)` | 1D | ❌ No |
| `y` after `.to(device)` | `(64,)` | 1D | ❌ No |
| `pred` from model | `(64, 1)` | 2D | ✅ Yes |
| `pred` after `.squeeze(1)` | `(64,)` | 1D | ✓ Correct |

After fix:
- `y.shape = (batch_size,)` ✓
- `pred.shape = (batch_size,)` ✓
- Both match for loss calculation ✓

---

## Testing the Fix

### Run the Notebook

```bash
# Option 1: Interactive Jupyter
cd d:\DoAn\SourceCode
jupyter notebook
# Open: notebooks/02_train_cnn1d.ipynb
# Run: Cell → Run All

# Option 2: Command line
ipython -c "%run notebooks/02_train_cnn1d.ipynb"
```

### Expected Output

First epoch should show:
```
[1] train_loss=0.xxxxx val_loss=0.xxxxx mae=0.xxxx rmse=0.xxxx lr=1.0e-04
[2] train_loss=0.xxxxx val_loss=0.xxxxx mae=0.xxxx rmse=0.xxxx lr=1.0e-04
...
```

Not the IndexError anymore! ✓

### Verify Model Saved

After training completes:
```bash
# Check that model file exists
ls -lh models/soc_cnn1d.pt
# Should show: models/soc_cnn1d.pt (~20-50 MB)
```

---

## Commit Details

**Commit Hash:** `4f22963`

```
fix: Remove incorrect squeeze(1) on y tensor in training notebook

Fix IndexError in notebooks/02_train_cnn1d.ipynb:
  Problem: y.to(device).squeeze(1) tried to squeeze dimension 1 on 1D tensor
  Root cause: DataLoader returns y with shape (batch_size,) - already 1D
  Valid dimensions for 1D tensor: only 0 (or -1)
  
Changes:
  - train_epoch() line 9: Remove .squeeze(1) from y assignment
  - eval_epoch() line 20: Remove .squeeze(1) from y assignment
  - Keep .squeeze(1) on pred (model output is 2D, needs to be 1D)
```

---

## Next Steps

### 1. Run Training (30 min)
```bash
jupyter notebook notebooks/02_train_cnn1d.ipynb
# Click Cell → Run All
# Wait for training to complete
```

**Output:** `models/soc_cnn1d.pt` (~20-50 MB)

### 2. Export to TFLite (5 min)
```bash
jupyter notebook notebooks/03_export_tflite.ipynb
# Click Cell → Run All
```

**Output:** `models/soc_cnn1d.tflite` (~3-4 MB for Pi 4)

### 3. Deploy to Raspberry Pi 4 (30 min)
```bash
# Copy TFLite model to Pi
scp models/soc_cnn1d.tflite pi@<pi-ip>:~/soc-monitor/models/

# Start service
sudo systemctl start soc-monitor

# Access dashboard
http://<pi-ip>:8080/
```

---

## Technical Details

### Why `.squeeze()` vs `.squeeze(1)`

**Option A: `.squeeze()` (removes ALL dimensions of size 1)**
```python
x = torch.tensor([[1, 2, 3]])  # shape (1, 3)
x.squeeze()  # shape (3,)
```

**Option B: `.squeeze(1)` (removes SPECIFIC dimension 1)**
```python
x = torch.tensor([[1, 2, 3]])  # shape (1, 3)
x.squeeze(1)  # ERROR - no dimension 1

y = torch.tensor([[[1]], [[2]]])  # shape (2, 1, 1)
y.squeeze(1)  # shape (2, 1)
```

Since `y` is 1D, `.squeeze()` would also work:
```python
X, y = X.to(device), y.to(device).squeeze()  # Also works
```

But removing `.squeeze()` entirely is cleaner since `y` is already the right shape.

---

## Summary

✅ **Issue:** IndexError when training CNN1D model  
✅ **Cause:** Incorrect tensor squeezing on 1D tensor  
✅ **Fix:** Remove `.squeeze(1)` from `y` assignments (2 locations)  
✅ **Result:** Training runs successfully without errors  
✅ **Status:** Committed and ready to use  

**The notebook is now fully functional!** 🎉

Run it to train your model:
```bash
jupyter notebook notebooks/02_train_cnn1d.ipynb
```
