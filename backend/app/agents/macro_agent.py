import json
from typing import Dict, Any
from .llm_helper import generate_text

# Standard macroeconomic context (simulated or real baseline)
MACRO_ENVIRONMENT = {
    "interest_rate": "5.25% - 5.50% (Federal Reserve Benchmark)",
    "inflation_rate": "2.8% (CPI Core YoY)",
    "gdp_growth": "2.2% (Real GDP annualized rate)",
    "dollar_index": "104.2 (DXY)",
    "oil_price": "$78.50 (WTI Crude)",
    "gold_price": "$2,350/oz",
    "vix": "13.8 (Volatility Index)"
}

def analyze(ticker: str, sector: str = "Technology") -> Dict[str, Any]:
    """
    Macro Economic Agent
    Analyzes how macroeconomic conditions (interest rates, GDP, oil, inflation) impact the stock's sector.
    """
    # Simple rule-based sector mapping
    sector_lower = sector.lower() if sector else ""
    
    if "tech" in sector_lower or "software" in sector_lower or "semiconductor" in sector_lower:
        sector_impact = "Neutral-to-Favorable"
        impact_details = "Growth-oriented tech firms remain sensitive to the cost of capital. However, secular trends like AI infrastructure demand outweigh short-term rate headwind concerns."
    elif "financial" in sector_lower or "bank" in sector_lower:
        sector_impact = "Favorable"
        impact_details = "Net interest margins remain supported by elevated yields, though slowing loan growth and credit delinquency tracking impose moderate caution."
    elif "energy" in sector_lower or "oil" in sector_lower:
        sector_impact = "Neutral"
        impact_details = "Energy returns are highly correlated with WTI Crude levels. Stability around $75-$80 provides stable operating cash flow but lacks breakout catalyst."
    elif "consumer" in sector_lower:
        sector_impact = "Neutral-to-Negative"
        impact_details = "Elevated inflation and interest rates continue to pressure household budgets, leading to consumer spending compression in discretionary categories."
    else:
        sector_impact = "Neutral"
        impact_details = "Steady GDP growth of 2.2% provides a stable backdrop for operations, balanced by high cost of debt refinancing."
        
    prompt = f"""
    You are the Macro Economic Agent. Analyze how the macro environment affects {ticker} in the {sector} sector:
    - Macro Environment: {json.dumps(MACRO_ENVIRONMENT)}
    - Sector impact direction: {sector_impact}
    - Baseline Sector logic: {impact_details}
    
    Provide an institutional macro analysis of sector performance, rate sensitivity, and macro economic transmission to this stock.
    Format your response as a JSON object with:
    "macro_data": {{ "interest_rate": "value", "inflation": "value", "gdp": "value", "vix": "value" }},
    "sector_impact": "Positive/Negative/Neutral",
    "explanation": "comprehensive narrative paragraph explaining the macro forces on this stock"
    """
    
    llm_output = generate_text(prompt, system_instruction="You are an international macroeconomist and institutional asset manager.")
    
    if llm_output:
        try:
            clean_output = llm_output.strip()
            if clean_output.startswith("```json"):
                clean_output = clean_output[7:]
            if clean_output.endswith("```"):
                clean_output = clean_output[:-3]
            data = json.loads(clean_output.strip())
            return data
        except Exception as e:
            print(f"Failed to parse Macro Agent LLM output: {e}")
            
    # Fallback simulation
    return {
        "macro_data": {
            "interest_rate": MACRO_ENVIRONMENT["interest_rate"],
            "inflation": MACRO_ENVIRONMENT["inflation_rate"],
            "gdp": MACRO_ENVIRONMENT["gdp_growth"],
            "vix": MACRO_ENVIRONMENT["vix"]
        },
        "sector_impact": "Positive" if "Favorable" in sector_impact else "Negative" if "Negative" in sector_impact else "Neutral",
        "explanation": f"For {ticker} ({sector}), the macro environment presents a {sector_impact.lower()} posture. {impact_details} Under the current Fed benchmark rate of 5.25%-5.50%, capital allocation efficiency remains paramount."
    }
