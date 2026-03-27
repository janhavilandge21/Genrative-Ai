"""
src/simulator.py
Real-time batch data simulator.
Generates a continuous stream of incoming batch records
and feeds them through the prediction pipeline.
"""

import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import preprocess_single
from src.model_trainer import predict_batch


def generate_random_batch(batch_num: int, inject_failure: bool = False) -> dict:
    """
    Generate a single random batch record.
    If inject_failure=True, intentionally create out-of-spec parameters.
    """
    if inject_failure:
        # Intentionally bad batch
        return {
            "batch_id":              f"SIM-{batch_num:04d}",
            "temperature_c":         round(random.uniform(85, 95), 2),   # Too hot
            "pressure_bar":          round(random.uniform(1.0, 1.5), 3), # Too low
            "mixing_time_min":       round(random.uniform(10, 20), 1),   # Too short
            "mixing_speed_rpm":      round(random.uniform(40, 60)),
            "humidity_pct":          round(random.uniform(70, 80), 1),   # Too humid
            "ph_level":              round(random.uniform(4.5, 5.2), 2), # Too acidic
            "particle_size_um":      round(random.uniform(350, 400), 1),
            "active_ingredient_pct": round(random.uniform(91, 93), 3),   # Low API
            "moisture_content_pct":  round(random.uniform(5.0, 6.0), 3), # High moisture
            "granulation_time_min":  round(random.uniform(8, 12), 1),
            "drying_temp_c":         round(random.uniform(35, 42), 1),
            "coating_thickness_um":  round(random.uniform(80, 100), 1),
            "tablet_hardness_n":     round(random.uniform(35, 50), 1),
            "dissolution_rate_pct":  round(random.uniform(50, 65), 2),   # Low dissolution
            "operator_experience_yr":round(random.uniform(0.5, 1.5), 1),
            "shift":                 2,                                    # Night shift
            "equipment_age_yr":      round(random.uniform(11, 15), 1),
            "raw_material_grade":    0,
        }
    else:
        # Normal batch (with small random variation)
        return {
            "batch_id":              f"SIM-{batch_num:04d}",
            "temperature_c":         round(random.normalvariate(65, 5), 2),
            "pressure_bar":          round(random.normalvariate(2.5, 0.3), 3),
            "mixing_time_min":       round(random.normalvariate(45, 5), 1),
            "mixing_speed_rpm":      round(random.normalvariate(120, 15)),
            "humidity_pct":          round(random.normalvariate(45, 5), 1),
            "ph_level":              round(random.normalvariate(6.8, 0.3), 2),
            "particle_size_um":      round(random.normalvariate(200, 20), 1),
            "active_ingredient_pct": round(random.normalvariate(98, 0.8), 3),
            "moisture_content_pct":  round(random.normalvariate(2.5, 0.4), 3),
            "granulation_time_min":  round(random.normalvariate(30, 4), 1),
            "drying_temp_c":         round(random.normalvariate(55, 4), 1),
            "coating_thickness_um":  round(random.normalvariate(150, 15), 1),
            "tablet_hardness_n":     round(random.normalvariate(80, 10), 1),
            "dissolution_rate_pct":  round(random.normalvariate(87, 4), 2),
            "operator_experience_yr":round(random.normalvariate(6, 2), 1),
            "shift":                 random.choice([0, 0, 0, 1, 2]),
            "equipment_age_yr":      round(random.normalvariate(4, 2), 1),
            "raw_material_grade":    random.choice([0, 1, 1]),
        }


def stream_batches(n_batches: int = 10, delay_seconds: float = 1.5):
    """
    Generator: yields batch results one by one for real-time simulation.

    Yields dicts with batch_data, prediction, timestamp, status.
    """
    for i in range(1, n_batches + 1):
        # Inject a failure every ~4th batch for demo
        inject_fail = (i % 4 == 0)
        batch = generate_random_batch(i, inject_failure=inject_fail)

        # Run through ML pipeline
        try:
            X_scaled, features = preprocess_single(batch)
            prediction = predict_batch(X_scaled)
        except Exception as e:
            prediction = {
                "failure_probability": 0.5,
                "failure_predicted": False,
                "is_anomaly": False,
                "anomaly_score": 0.0,
            }

        # Determine alert level
        fp = prediction["failure_probability"]
        if fp >= 0.75 or prediction["is_anomaly"]:
            status = "CRITICAL"
            color  = "red"
        elif fp >= 0.45:
            status = "WARNING"
            color  = "orange"
        else:
            status = "NORMAL"
            color  = "green"

        yield {
            "batch_num":   i,
            "batch_data":  batch,
            "prediction":  prediction,
            "status":      status,
            "color":       color,
            "timestamp":   datetime.now().strftime("%H:%M:%S"),
        }

        time.sleep(delay_seconds)
