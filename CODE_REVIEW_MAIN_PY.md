# CODE REVIEW: src/main.py — Task 19

**Status:** ⭐⭐⭐⭐ (4/5) — Solid structure, 2 critical fixes needed

---

## 1. CRITICAL ISSUES (Must fix before running)

### Issue 1: Inference window shape WRONG
**Location:** Lines 281-286

```python
window = np.array([
    [v_mean, i_mean, tmp_mean, spd_mean]  # ← Shape (1, 4) ✗ WRONG
], dtype=np.float32)
```

**Problem:**
- Model expects shape: `(batch, window_size, 4)` = `(1, 60, 4)`
- This gives: `(1, 4)` — missing window dimension!
- `soc_inference.predict()` will fail shape validation

**Fix:** Create proper 60-sample window
```python
# Option A: Pad current sample (simple but naive)
sample = np.array([v_mean, i_mean, tmp_mean, spd_mean], dtype=np.float32)
window = np.tile(sample, (60, 1))  # (60, 4)
window = window[np.newaxis, :]  # (1, 60, 4) ✓

# Option B: Use actual 10 samples, pad to 60
if len(v_arr) < 60:
    v_padded = np.pad(v_arr, (60-len(v_arr), 0), 'edge')
    # ... same for i, spd, tmp
    window = np.stack([v_padded, i_padded, spd_padded, tmp_padded], axis=1)
else:
    window = np.stack([v_arr[-60:], i_arr[-60:], spd_arr[-60:], tmp_arr[-60:]], axis=1)
window = window[np.newaxis, :]  # (1, 60, 4) ✓
```

**Impact:** 🔴 SoC#3 predictions invalid until fixed

---

### Issue 2: Missing normalization
**Location:** Line 277 (TODO comment)

**Problem:**
- Model trained on normalized data (minmax or zscore)
- Feeding raw values → predictions garbage
- All SoC#3 results will be nonsensical

**Fix:** Apply normalization from configs/model.yaml
```python
import yaml
from src.preprocessing.normalize import normalize

# Load normalization config
with open("configs/model.yaml") as f:
    model_config = yaml.safe_load(f)
norm_params = model_config["normalization"]

# Normalize each feature
window_normalized = normalize(
    window,
    method=norm_params["method"],  # "minmax" or "zscore"
    bounds={
        "pack_voltage_v": norm_params["pack_voltage_v"],
        "pack_current_a": norm_params["pack_current_a"],
        "temp_c": norm_params["temp_c"],
        "speed_kmh": norm_params["speed_kmh"],
    }
)

# Inference with normalized data
soc_model, soh = soc_inference.predict(window_normalized)
```

**Impact:** 🔴 All model predictions invalid until fixed

---

## 2. HIGH PRIORITY ISSUES

### Issue 3: Buffer size too small for range_estimator
**Location:** Line 37 & 295-296

```python
RING_BUFFER_SIZE = 10  # 1 second at 10Hz

# Later:
range_estimator.update_and_estimate(
    speed_window=spd_arr[-60:] if len(spd_arr) >= 60 else spd_arr,  # ← never gets 60!
)
```

**Problem:**
- Buffer only keeps 10 samples (1 second)
- Range estimator expects 60 samples (6 seconds history)
- Takes 6 seconds before range becomes accurate

**Fix:** Increase RING_BUFFER_SIZE
```python
RING_BUFFER_SIZE = 60  # Keep 6 seconds of data at 10Hz
```

**Tradeoff:** Slightly more memory (60 × 4 floats × 4 buffers = ~1KB), better range accuracy

**Impact:** 🟡 Range estimates inaccurate for first 6 seconds

---

### Issue 4: Lock held during inference (30-50ms)
**Location:** Line 261-303

```python
if tick % INFERENCE_EVERY_N_TICK == 0:
    try:
        with lock:  # ← Lock held for entire inference block
            v_arr, i_arr, spd_arr, tmp_arr = state.get_buffers_as_arrays()
            # ... 30-50ms inference and range_calc ...
            soc_model, soh = soc_inference.predict(window)  # ← Slow operation
            range_km, wh_per_km = range_estimator.update_and_estimate(...)  # ← Slow
```

**Problem:**
- Lock blocks web server /api/state requests for 30-50ms
- On Pi, causes latency spikes

**Fix:** Acquire lock only for I/O, not computation
```python
# Read state (fast, locked)
with lock:
    v_arr, i_arr, spd_arr, tmp_arr = state.get_buffers_as_arrays()

# Inference (slow, unlocked)
soc_model, soh = soc_inference.predict(window)
range_km, wh_per_km = range_estimator.update_and_estimate(...)

# Write results (fast, locked)
with lock:
    state.soc_model = soc_model
    state.soh = soh
    state.range_km = range_km
    state.wh_per_km = wh_per_km
```

**Impact:** 🟡 Web server latency spikes every second

---

## 3. MEDIUM PRIORITY ISSUES

### Issue 5: CAN reconnect logic missing
**Location:** Line 220-256

```python
try:
    frames = can_reader.read_frames()  # ← Raises exception if disconnected
except Exception as e:
    logger.warning(f"CAN read error: {e}")  # ← Logs and continues
    # No recovery attempt!
```

**Problem:**
- If Waveshare unplugged: exception logged but loop continues
- No automatic reconnect
- Silent failure until manual intervention

**Fix:** Add reconnect logic
```python
except Exception as e:
    logger.warning(f"CAN read error: {e}")
    try:
        logger.info("Attempting to reconnect to Waveshare...")
        can_reader.disconnect()
        can_reader.connect()
    except Exception as e2:
        logger.error(f"Reconnect failed: {e2}")
```

**Impact:** 🟡 Graceful degradation without automatic recovery

---

### Issue 6: Inefficient FPS tracking
**Location:** Line 214, 320-322

```python
tick_times = []  # ← Standard list
# ...
tick_times.append(elapsed)
if len(tick_times) > TICK_HZ:
    tick_times.pop(0)  # ← O(n) operation, inefficient
```

**Problem:**
- `pop(0)` on list is O(n) in Python
- Every tick: 1ms of FPS calculation

**Fix:** Use collections.deque
```python
from collections import deque

tick_times = deque(maxlen=TICK_HZ)  # ← Fixed-size, O(1) pop
# ...
tick_times.append(elapsed)
# No need for length check!
with lock:
    state.fps_actual = TICK_HZ / sum(tick_times) if sum(tick_times) > 0 else 0
```

**Impact:** 🟢 Negligible, <1% performance improvement

---

## 4. POSITIVE ASPECTS ✅

- ✅ **Architecture compliance:** Matches service_architecture.md §4
- ✅ **Code style:** Clear docstrings, good naming, logical sections
- ✅ **Type hints:** Function signatures have proper types
- ✅ **Error handling:** try/except for CAN read and inference
- ✅ **Threading:** Proper lock usage (mostly)
- ✅ **Timing:** Correct use of time.monotonic() and sleep adjustment
- ✅ **Separation of concerns:** SharedState, init_system(), main_loop(), start_web_server()

---

## 5. TESTING CHECKLIST

Before deployment:
- [ ] Fix Issue 1: Inference window shape (1, 60, 4)
- [ ] Fix Issue 2: Apply normalization to input
- [ ] Fix Issue 3: RING_BUFFER_SIZE = 60
- [ ] Fix Issue 4: Move inference outside lock
- [ ] Test: SoC#1 updates every tick
- [ ] Test: SoC#2 (CC) tracks BMS roughly
- [ ] Test: SoC#3 produces reasonable values (after fixes)
- [ ] Test: Range estimation after 6 seconds of data
- [ ] Test: Web /api/state responds in <100ms
- [ ] Test: FPS stays at 10Hz (±5%)

---

## 6. SUMMARY

| Aspect | Status | Notes |
|--------|--------|-------|
| Architecture | ✅ Good | Per spec, single-thread 10Hz |
| Code structure | ✅ Good | Clear, modular |
| Critical issues | 🔴 2 | Inference shape + normalization |
| High priority | 🟡 2 | Buffer size + lock timing |
| Logic | ⚠️ Mostly OK | Ring buffer FIFO correct, CC reset logic OK |
| Performance | ✅ OK | Should maintain 10Hz, inference fits in 1Hz slot |
| Error handling | ⚠️ Basic | No reconnect logic, Flask import not validated |

**Recommendation:** ⏳ **APPROVED with mandatory fixes** — Fix Issues 1-2 before running on Pi.

---

## Implementation Plan

**3 follow-up commits needed:**

1. **fix: main.py inference window shape + normalization**
   - Create proper (1, 60, 4) window
   - Apply minmax/zscore normalization
   
2. **fix: main.py buffer size + lock timing**
   - RING_BUFFER_SIZE = 60
   - Move inference outside critical section

3. **fix: main.py reconnect logic + FPS tracking**
   - Add CAN reconnect attempt
   - Use collections.deque for FPS

Total effort: ~1-2 hours

