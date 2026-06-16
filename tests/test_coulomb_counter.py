"""
Test Coulomb Counter Module — update, reset, bounds, sign convention.

Smoke tests để kiểm tra:
- Khởi tạo CoulombCounter
- Update với dòng xả (I > 0): SoC giảm
- Update với dòng sạc (I < 0): SoC tăng
- Bounds clamping [0, 100]
- Reset functionality
- should_reset condition
- dt calculation (thời gian thực)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.coulomb_counter.counter import CoulombCounter


def test_init_default():
    """Khởi tạo CoulombCounter với giá trị mặc định."""
    cc = CoulombCounter()

    assert cc.soc_cc == 100.0, f"Expected SoC=100%, got {cc.soc_cc}"
    assert cc.capacity_ah > 0, f"Capacity should be positive"

    print("[OK] Default initialization: SoC=100%")


def test_init_custom():
    """Khởi tạo CoulombCounter với custom capacity và SoC."""
    cc = CoulombCounter(capacity_ah=50.0, initial_soc=80.0)

    assert cc.soc_cc == 80.0, f"Expected SoC=80%, got {cc.soc_cc}"
    assert cc.capacity_ah == 50.0, f"Expected capacity=50Ah, got {cc.capacity_ah}"

    print("[OK] Custom initialization: SoC=80%, capacity=50Ah")


def test_update_discharge():
    """Kiểm tra SoC giảm khi dòng xả (I > 0)."""
    cc = CoulombCounter(capacity_ah=30.0, initial_soc=100.0)

    # Xả 15A trong 1 giây
    # DeltaSoC = -(15 × 1 / 3600) / 30 × 100 = -0.0139%
    initial_soc = cc.soc_cc
    soc_after = cc.update(current_a=15.0, dt=1.0)

    assert soc_after < initial_soc, \
        f"SoC should decrease during discharge, {initial_soc} -> {soc_after}"
    assert abs(soc_after - initial_soc) < 0.1, \
        f"1 second discharge should be small, got {initial_soc - soc_after}%"

    print(f"[OK] Discharge: SoC {initial_soc:.1f}% -> {soc_after:.1f}%")


def test_update_charge():
    """Kiểm tra SoC tăng khi dòng sạc (I < 0)."""
    cc = CoulombCounter(capacity_ah=30.0, initial_soc=50.0)

    initial_soc = cc.soc_cc
    soc_after = cc.update(current_a=-15.0, dt=1.0)

    assert soc_after > initial_soc, \
        f"SoC should increase during charge, {initial_soc} -> {soc_after}"

    print(f"[OK] Charge: SoC {initial_soc:.1f}% -> {soc_after:.1f}%")


def test_update_zero_current():
    """Kiểm tra SoC không thay đổi khi I=0."""
    cc = CoulombCounter(capacity_ah=30.0, initial_soc=75.0)

    initial_soc = cc.soc_cc
    soc_after = cc.update(current_a=0.0, dt=1.0)

    assert soc_after == initial_soc, \
        f"SoC should not change with zero current, got {initial_soc} -> {soc_after}"

    print("[OK] Zero current: SoC unchanged")


def test_update_zero_dt():
    """Kiểm tra SoC không thay đổi khi dt=0."""
    cc = CoulombCounter(capacity_ah=30.0, initial_soc=75.0)

    initial_soc = cc.soc_cc
    soc_after = cc.update(current_a=15.0, dt=0.0)

    assert soc_after == initial_soc, \
        f"SoC should not change with zero dt, got {initial_soc} -> {soc_after}"

    print("[OK] Zero dt: SoC unchanged")


def test_bounds_lower():
    """Kiểm tra SoC không giảm dưới 0%."""
    cc = CoulombCounter(capacity_ah=30.0, initial_soc=0.1)

    # Xả lớn để SoC ngoài range
    soc_after = cc.update(current_a=100.0, dt=10.0)

    assert soc_after >= 0.0, f"SoC should not go below 0%, got {soc_after}"
    assert soc_after == 0.0, f"SoC should be clamped to 0%, got {soc_after}"

    print("[OK] Lower bound: SoC clamped at 0%")


def test_bounds_upper():
    """Kiểm tra SoC không vượt quá 100%."""
    cc = CoulombCounter(capacity_ah=30.0, initial_soc=99.9)

    # Sạc lớn để SoC ngoài range
    soc_after = cc.update(current_a=-100.0, dt=10.0)

    assert soc_after <= 100.0, f"SoC should not exceed 100%, got {soc_after}"
    assert soc_after == 100.0, f"SoC should be clamped to 100%, got {soc_after}"

    print("[OK] Upper bound: SoC clamped at 100%")


def test_reset():
    """Kiểm tra reset SoC về giá trị mới."""
    cc = CoulombCounter(capacity_ah=30.0, initial_soc=50.0)

    # Update SoC
    cc.update(current_a=15.0, dt=10.0)
    soc_before_reset = cc.soc_cc
    assert soc_before_reset < 50.0

    # Reset
    cc.reset(new_soc=85.0)
    assert cc.soc_cc == 85.0, \
        f"Reset should set SoC to 85%, got {cc.soc_cc}"

    print(f"[OK] Reset: SoC {soc_before_reset:.1f}% -> 85.0%")


def test_should_reset_below_threshold():
    """Kiểm tra should_reset khi SoC < ngưỡng sạc đầy."""
    cc = CoulombCounter(capacity_ah=30.0, initial_soc=80.0)

    # SoC < 98% (ngưỡng mặc định)
    assert not cc.should_reset(soc_bms=80.0)

    print("[OK] should_reset: False when SoC < 98%")


def test_should_reset_at_threshold():
    """Kiểm tra should_reset khi SoC >= ngưỡng sạc đầy."""
    cc = CoulombCounter(capacity_ah=30.0, initial_soc=80.0)

    # SoC >= 98% (ngưỡng mặc định) -> nên reset
    assert cc.should_reset(soc_bms=98.0)
    assert cc.should_reset(soc_bms=100.0)

    print("[OK] should_reset: True when SoC >= 98%")


def test_dt_realistic():
    """Kiểm tra dt = 0.143s (7Hz sample rate) như dataset Evo200."""
    cc = CoulombCounter(capacity_ah=30.65, initial_soc=80.0)

    # Dataset Evo200 có ~7Hz (dt approx 0.143s), không phải 10Hz (0.1s)
    # Xả 10A trong ~40 samples (6 giây)
    soc_start = cc.soc_cc

    for _ in range(40):
        cc.update(current_a=10.0, dt=0.143)

    soc_end = cc.soc_cc
    delta_soc = soc_start - soc_end

    # DeltaSoC = -(10 × 0.143 × 40 / 3600) / 30.65 × 100
    #      = -(57.2 / 3600) / 30.65 × 100 approx 0.052%
    expected_delta = (10.0 * 0.143 * 40 / 3600.0) / 30.65 * 100.0

    assert abs(delta_soc - expected_delta) < 0.01, \
        f"Expected DeltaSoCapprox{expected_delta:.3f}%, got {delta_soc:.3f}%"

    print(f"[OK] dt=0.143s (7Hz): DeltaSoC={delta_soc:.3f}% as expected")


def test_coulomb_formula_sign():
    """Kiểm tra công thức Coulomb counting sign convention."""
    cc = CoulombCounter(capacity_ah=30.0, initial_soc=100.0)

    # Xả 1A trong 3600 giây = 1 Ah
    # DeltaSoC = -(1 × 3600 / 3600) / 30 × 100 = -100/30 = -3.33%
    soc_before = cc.soc_cc
    soc_after = cc.update(current_a=1.0, dt=3600.0)

    expected_delta = -(1.0 * 3600.0 / 3600.0) / 30.0 * 100.0  # -3.33%
    actual_delta = soc_after - soc_before

    assert abs(actual_delta - expected_delta) < 0.01, \
        f"Expected DeltaSoC={expected_delta:.3f}%, got {actual_delta:.3f}%"

    print(f"[OK] Coulomb formula: DeltaSoC = {actual_delta:.3f}% for 1Ah discharge")


def test_drift_over_time():
    """Kiểm tra drift tích lũy qua nhiều cycles."""
    cc = CoulombCounter(capacity_ah=30.0, initial_soc=100.0)

    # Simulated discharge/charge cycle
    # Cycle 1: xả 10A × 10s
    for _ in range(10):
        cc.update(current_a=10.0, dt=1.0)

    soc_after_discharge = cc.soc_cc
    assert soc_after_discharge < 100.0

    # Charge lại 10A × 10s
    for _ in range(10):
        cc.update(current_a=-10.0, dt=1.0)

    soc_after_charge = cc.soc_cc
    assert soc_after_charge >= 99.9, \
        f"After balanced discharge-charge, SoC should return to ~100%, got {soc_after_charge}"

    print(f"[OK] Drift: Balanced cycle returns to {soc_after_charge:.1f}%")


def test_reset_to_anchor():
    """Kiểm tra reset được dùng để re-anchor từ BMS khi sạc đầy."""
    cc = CoulombCounter(capacity_ah=30.0, initial_soc=100.0)

    # Simulated: xả -> drift tích lũy -> sạc đầy detected -> reset
    for _ in range(20):
        cc.update(current_a=10.0, dt=1.0)

    soc_with_drift = cc.soc_cc  # < 100% vì drift
    assert soc_with_drift < 100.0

    # BMS báo đã sạc đầy (100%) -> reset Coulomb Counter
    if cc.should_reset(soc_bms=100.0):
        cc.reset(new_soc=100.0)

    assert cc.soc_cc == 100.0, "After reset, SoC should match BMS"

    print(f"[OK] Anchor reset: Coulomb Counter {soc_with_drift:.1f}% -> 100% (re-anchored to BMS)")


if __name__ == "__main__":
    print("Testing Coulomb Counter Module...")
    print()

    test_init_default()
    test_init_custom()
    test_update_discharge()
    test_update_charge()
    test_update_zero_current()
    test_update_zero_dt()
    test_bounds_lower()
    test_bounds_upper()
    test_reset()
    test_should_reset_below_threshold()
    test_should_reset_at_threshold()
    test_dt_realistic()
    test_coulomb_formula_sign()
    test_drift_over_time()
    test_reset_to_anchor()

    print()
    print("All coulomb counter smoke tests passed!")
