import json
from typing import Dict, Any
from .llm_helper import generate_text

def analyze(ticker: str, tech_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Technical Analysis Agent
    Evaluates RSI, MACD, EMAs/SMAs, Support & Resistance, and Trend.
    """
    if "error" in tech_data:
        return {
            "score": 5.0,
            "trend": "Sideways",
            "rsi_signal": "Neutral",
            "macd_signal": "Neutral",
            "summary": "Technical data unavailable.",
            "signals": [],
            "indicators": tech_data
        }
        
    rsi = tech_data["rsi"]
    macd_hist = tech_data["macd_hist"]
    trend = tech_data["trend"]
    current_price = tech_data["current_price"]
    sma_20 = tech_data["sma_20"]
    sma_50 = tech_data["sma_50"]
    sma_200 = tech_data["sma_200"]
    
    # Heuristics for score
    score = 5.0
    
    # RSI score impact
    if rsi < 30: # Oversold (Bullish reversion)
        score += 1.5
        rsi_signal = "Oversold (Buy Signal)"
    elif rsi > 70: # Overbought (Bearish exhaustion)
        score -= 1.0
        rsi_signal = "Overbought (Sell Signal)"
    elif 50 <= rsi <= 70: # Strong momentum
        score += 1.0
        rsi_signal = "Bullish Momentum"
    else: # Weak bearish momentum
        score -= 0.5
        rsi_signal = "Bearish Momentum"
        
    # MACD score impact
    if macd_hist > 0:
        score += 1.0
        macd_signal = "Bullish Crossover"
    else:
        score -= 1.0
        macd_signal = "Bearish Crossover"
        
    # Trend alignment
    if trend == "Bullish":
        score += 1.5
    elif trend == "Bearish":
        score -= 1.5
        
    score = max(1.0, min(10.0, score))
    
    # Assemble signals list
    signals = []
    signals.append(f"RSI is currently {rsi:.1f}, representing a {rsi_signal} state.")
    signals.append(f"MACD histogram is {'positive' if macd_hist > 0 else 'negative'} at {macd_hist:.4f}, demonstrating {macd_signal.lower()}.")
    signals.append(f"Price is at ${current_price:.2f}, relative to 20-day SMA of ${sma_20:.2f} and 50-day SMA of ${sma_50:.2f}.")
    signals.append(f"Key support found at ${tech_data['support']:.2f} and resistance overhead at ${tech_data['resistance']:.2f}.")
    
    # Try LLM for qualitative report
    prompt = f"""
    You are the Technical Analysis Agent. Analyze the stock technical charts for {ticker}:
    - Current Price: {current_price}
    - RSI (14): {rsi:.1f}
    - MACD Histogram: {macd_hist:.4f}
    - 20-day SMA: {sma_20:.2f}
    - 50-day SMA: {sma_50:.2f}
    - 200-day SMA: {sma_200:.2f}
    - Support: {tech_data['support']:.2f}
    - Resistance: {tech_data['resistance']:.2f}
    - Price Trend: {trend}
    
    Provide an institutional-grade technical analysis report detailing the price action, key trendline patterns, momentum indicators, and support/resistance dynamics.
    Format your response as a JSON object with:
    "score": float between 1.0 and 10.0,
    "trend": "Bullish/Bearish/Sideways",
    "rsi_evaluation": "RSI evaluation sentence",
    "macd_evaluation": "MACD evaluation sentence",
    "summary": "overall technical analysis narrative paragraph"
    """
    
    llm_output = generate_text(prompt, system_instruction="You are an expert CMT (Chartered Market Technician) and algorithmic trading system.")
    
    if llm_output:
        try:
            clean_output = llm_output.strip()
            if clean_output.startswith("```json"):
                clean_output = clean_output[7:]
            if clean_output.endswith("```"):
                clean_output = clean_output[:-3]
            data = json.loads(clean_output.strip())
            # Inject computed details
            data["indicators"] = tech_data
            data["signals"] = signals
            data["score"] = float(data.get("score", score))
            return data
        except Exception as e:
            print(f"Failed to parse Technical Agent LLM output: {e}")
            
    # Fallback simulation
    return {
        "score": round(score, 1),
        "trend": trend,
        "rsi_evaluation": f"RSI is at {rsi:.1f}, indicating {rsi_signal.lower()} conditions.",
        "macd_evaluation": f"MACD is {'bullish' if macd_hist > 0 else 'bearish'}, with the histogram expanding at {macd_hist:.4f}.",
        "summary": f"Technical profile is {trend} (Score: {score:.1f}/10). The stock shows strong consolidation. Immediate support resides at ${tech_data['support']:.2f}, while resistance at ${tech_data['resistance']:.2f} remains the primary bullish target.",
        "indicators": tech_data,
        "signals": signals
    }
