"""
src/data_generator.py
Generates a realistic pharmaceutical manufacturing dataset.
Simulates batch records for a tablet/capsule production line.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Output path ───────────────────────────────────────────────────────────────
DATA_DIR  = Path(__file__).parent.parent / "data"
DATA_FILE = DATA_DIR / "pharma_batches.csv"


def generate_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generate a realistic pharma manufacturing dataset.

    Features (process parameters):
    - temperature_c        : Reactor/mixing temperature (°C)
    - pressure_bar         : Vessel pressure (bar)
    - mixing_time_min      : Duration of mixing step (minutes)
    - mixing_speed_rpm     : Mixing speed (RPM)
    - humidity_pct         : Relative humidity in room (%)
    - ph_level             : pH of the solution/slurry
    - particle_size_um     : Average particle size (microns)
    - active_ingredient_pct: % active pharmaceutical ingredient (API)
    - moisture_content_pct : Moisture content of blend (%)
    - granulation_time_min : Time for granulation step (minutes)
    - drying_temp_c        : Drying oven temperature (°C)
    - coating_thickness_um : Tablet coating thickness (microns)
    - tablet_hardness_n    : Tablet hardness in Newtons
    - dissolution_rate_pct : % dissolution at 30 min
    - operator_experience_yr: Years of experience of operator
    - shift                : Manufacturing shift (0=Day, 1=Evening, 2=Night)
    - equipment_age_yr     : Age of manufacturing equipment (years)
    - raw_material_grade   : Grade of raw material (0=Standard, 1=Premium)

    Target:
    - batch_failed : 1 = batch failed QC, 0 = passed
    """

    n = n_samples

    # ── Process parameters (normally distributed around optimal values) ───────
    temperature_c         = np.random.normal(65,  8,  n).clip(40,  95)
    pressure_bar          = np.random.normal(2.5, 0.5, n).clip(1.0, 5.0)
    mixing_time_min       = np.random.normal(45,  10, n).clip(15,  90)
    mixing_speed_rpm      = np.random.normal(120, 25, n).clip(50,  250)
    humidity_pct          = np.random.normal(45,  10, n).clip(20,  80)
    ph_level              = np.random.normal(6.8, 0.6, n).clip(4.5, 9.0)
    particle_size_um      = np.random.normal(200, 40, n).clip(80,  400)
    active_ingredient_pct = np.random.normal(98,  1.5, n).clip(92,  102)
    moisture_content_pct  = np.random.normal(2.5, 0.8, n).clip(0.5, 6.0)
    granulation_time_min  = np.random.normal(30,  8,  n).clip(10,  60)
    drying_temp_c         = np.random.normal(55,  7,  n).clip(35,  80)
    coating_thickness_um  = np.random.normal(150, 25, n).clip(80,  250)
    tablet_hardness_n     = np.random.normal(80,  15, n).clip(40,  140)
    dissolution_rate_pct  = np.random.normal(85,  8,  n).clip(50,  100)
    operator_experience_yr = np.random.exponential(5, n).clip(0.5, 20)
    shift                 = np.random.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2])
    equipment_age_yr      = np.random.exponential(4, n).clip(0.5, 15)
    raw_material_grade    = np.random.choice([0, 1], n, p=[0.6, 0.4])

    # ── Realistic failure logic (domain-driven) ───────────────────────────────
    # Failure probability driven by out-of-spec parameters
    fail_score = np.zeros(n)

    # Temperature deviation from optimal (65°C ± 10)
    fail_score += np.where(temperature_c < 50, 2.0, 0)
    fail_score += np.where(temperature_c > 80, 2.5, 0)

    # High moisture → sticking, capping in tablets
    fail_score += np.where(moisture_content_pct > 4.5, 2.0, 0)
    fail_score += np.where(moisture_content_pct > 5.5, 1.5, 0)

    # pH out of range → API degradation
    fail_score += np.where(ph_level < 5.5, 2.0, 0)
    fail_score += np.where(ph_level > 8.0, 1.5, 0)

    # API content out of spec (98±2%)
    fail_score += np.where(active_ingredient_pct < 94, 3.0, 0)
    fail_score += np.where(active_ingredient_pct > 101, 1.5, 0)

    # Low dissolution — bioavailability failure
    fail_score += np.where(dissolution_rate_pct < 70, 2.5, 0)

    # High humidity → moisture absorption
    fail_score += np.where(humidity_pct > 65, 1.5, 0)

    # Low mixing → poor blend uniformity
    fail_score += np.where(mixing_time_min < 25, 1.5, 0)
    fail_score += np.where(mixing_speed_rpm < 70, 1.0, 0)

    # Night shift & inexperienced operator → more errors
    fail_score += np.where(shift == 2, 0.8, 0)
    fail_score += np.where(operator_experience_yr < 2, 1.0, 0)

    # Old equipment
    fail_score += np.where(equipment_age_yr > 10, 1.0, 0)

    # Low tablet hardness → friability
    fail_score += np.where(tablet_hardness_n < 55, 1.5, 0)

    # Standard raw material grade
    fail_score += np.where(raw_material_grade == 0, 0.5, 0)

    # Convert score to probability
    fail_prob = 1 / (1 + np.exp(-(fail_score - 3.5)))  # sigmoid centered at 3.5

    # Add noise
    fail_prob = np.clip(fail_prob + np.random.normal(0, 0.05, n), 0.02, 0.98)

    batch_failed = (np.random.rand(n) < fail_prob).astype(int)

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame({
        "batch_id":              [f"BATCH-{str(i+1000).zfill(5)}" for i in range(n)],
        "temperature_c":         np.round(temperature_c, 2),
        "pressure_bar":          np.round(pressure_bar, 3),
        "mixing_time_min":       np.round(mixing_time_min, 1),
        "mixing_speed_rpm":      np.round(mixing_speed_rpm, 0).astype(int),
        "humidity_pct":          np.round(humidity_pct, 1),
        "ph_level":              np.round(ph_level, 2),
        "particle_size_um":      np.round(particle_size_um, 1),
        "active_ingredient_pct": np.round(active_ingredient_pct, 3),
        "moisture_content_pct":  np.round(moisture_content_pct, 3),
        "granulation_time_min":  np.round(granulation_time_min, 1),
        "drying_temp_c":         np.round(drying_temp_c, 1),
        "coating_thickness_um":  np.round(coating_thickness_um, 1),
        "tablet_hardness_n":     np.round(tablet_hardness_n, 1),
        "dissolution_rate_pct":  np.round(dissolution_rate_pct, 2),
        "operator_experience_yr":np.round(operator_experience_yr, 1),
        "shift":                 shift,
        "equipment_age_yr":      np.round(equipment_age_yr, 1),
        "raw_material_grade":    raw_material_grade,
        "batch_failed":          batch_failed,
    })

    print(f"✅ Dataset generated: {len(df)} batches | "
          f"Failure rate: {df['batch_failed'].mean()*100:.1f}%")
    return df


def save_dataset(df: pd.DataFrame) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"💾 Saved to: {DATA_FILE}")
    return str(DATA_FILE)


if __name__ == "__main__":
    df = generate_dataset(2000)
    save_dataset(df)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Failure rate: {df['batch_failed'].mean()*100:.1f}%")
