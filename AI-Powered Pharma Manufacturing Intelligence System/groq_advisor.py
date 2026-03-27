"""
src/groq_advisor.py
Groq LLM integration for human-readable batch analysis and recommendations.
Uses LLaMA3-70b via Groq API for ultra-fast inference.
"""

import os
import json
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── System prompt (as specified in requirements) ──────────────────────────────
SYSTEM_PROMPT = """You are an AI expert in pharmaceutical manufacturing. 
Analyze the batch data and explain why failure might happen and how to fix it in simple business language.

Your response must follow this structure:
1. RISK SUMMARY (2-3 sentences, plain language for managers)
2. ROOT CAUSE ANALYSIS (bullet points — what process parameters are out of spec and why they matter)
3. IMMEDIATE ACTIONS (3-5 specific, actionable steps the production team should take RIGHT NOW)
4. PREVENTIVE MEASURES (2-3 long-term recommendations to avoid recurrence)
5. QUALITY IMPACT (brief note on what product quality issue this batch might cause if released)

Be concise, specific, and use pharma manufacturing terminology where appropriate.
Avoid jargon that non-technical managers wouldn't understand.
Always end with a clear GO / NO-GO / HOLD recommendation for the batch."""


def analyze_batch_with_groq(
    batch_data: dict,
    prediction: dict,
    shap_top: dict,
    feature_importances: dict | None = None,
) -> str:
    """
    Call Groq LLM to generate a human-readable batch analysis report.

    Args:
        batch_data:          Raw batch parameters dict
        prediction:          Output from model_trainer.predict_batch()
        shap_top:            Top SHAP values (feature -> contribution)
        feature_importances: Overall feature importances from training

    Returns:
        Formatted string with AI recommendations
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key == "gsk_your_groq_key_here":
        return (
            "⚠️ GROQ_API_KEY not configured. "
            "Add your key to the .env file to enable AI recommendations.\n"
            "Get a free key at: https://console.groq.com"
        )

    client = Groq(api_key=api_key)

    # Format top risk factors for the prompt
    top_factors = list(shap_top.items())[:8]  # Top 8 features
    factors_str = "\n".join(
        f"  - {feat}: impact = {val:+.3f} ({'↑ increases failure risk' if val > 0 else '↓ decreases failure risk'})"
        for feat, val in top_factors
    )

    # Build the user message
    fail_pct = prediction["failure_probability"] * 100
    anomaly_status = "YES - ANOMALOUS BATCH" if prediction["is_anomaly"] else "No anomaly detected"

    user_message = f"""PHARMA BATCH ANALYSIS REQUEST

BATCH PARAMETERS:
{json.dumps({k: v for k, v in batch_data.items() if k != 'batch_id'}, indent=2)}

ML MODEL RESULTS:
- Failure Probability: {fail_pct:.1f}%
- Failure Predicted: {'YES ❌' if prediction['failure_predicted'] else 'NO ✅'}
- Anomaly Detection: {anomaly_status}
- Anomaly Score: {prediction['anomaly_score']:.4f} (more negative = more anomalous)

TOP RISK FACTORS (SHAP Analysis — what is driving the failure prediction):
{factors_str}

REFERENCE RANGES (normal operating limits):
- Temperature: 55–75°C (optimal: 65°C)
- Pressure: 2.0–3.0 bar
- Mixing Time: 35–55 minutes
- pH Level: 6.2–7.4
- Moisture Content: <3.5%
- Active Ingredient: 96–100%
- Dissolution Rate: >80%
- Humidity: <60%
- Tablet Hardness: 65–120 N

Please provide your expert analysis and recommendations."""

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return f"⚠️ AI analysis temporarily unavailable: {str(e)}"


def get_realtime_alert(batch_data: dict, severity: str) -> str:
    """
    Generate a short real-time alert message for streaming simulation.
    Designed to be fast (<500ms) by using a smaller model and fewer tokens.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key == "gsk_your_groq_key_here":
        return f"[{severity.upper()} ALERT] Batch requires immediate attention."

    client = Groq(api_key=api_key)

    prompt = f"""You are a pharma QC alert system. Generate a ONE-LINE alert message (max 30 words) for this batch:
Severity: {severity}
Key parameters: {json.dumps({k: v for k, v in list(batch_data.items())[:6]}, indent=None)}
Alert message:"""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",   # Faster model for real-time alerts
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=60,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return f"[{severity.upper()}] Batch anomaly detected — review parameters immediately."
