import json
from typing import Dict, Any, List
from .llm_helper import generate_text

# Simulated insider activity for popular stocks
POPULAR_INSIDER_DATA = {
    "AAPL": {
        "activity": "Net Institutional Selling",
        "institutional_change_pct": -2.8,
        "recent_trades": [
            {"date": "2026-06-15", "insider": "Cook Timothy D (CEO)", "type": "Sale (Option Exercise)", "shares": 120000, "price": 190.50},
            {"date": "2026-05-10", "insider": "Levinson Arthur D (Director)", "type": "Sale", "shares": 35000, "price": 185.20}
        ],
        "summary": "Berkshire Hathaway trimmed its Apple holdings by a minor percentage last quarter, offset by buying from index funds. CEO Tim Cook executed scheduled sales under an active 10b5-1 plan."
    },
    "NVDA": {
        "activity": "Neutral-to-Selling",
        "institutional_change_pct": 1.4,
        "recent_trades": [
            {"date": "2026-06-20", "insider": "Huang Jen Hsun (CEO)", "type": "Sale (Rule 10b5-1)", "shares": 240000, "price": 125.40},
            {"date": "2026-05-15", "insider": "Stevens Colette (Director)", "type": "Sale", "shares": 10000, "price": 118.80}
        ],
        "summary": "CEO Jensen Huang continues regular, pre-scheduled share disposals under a 10b5-1 plan. Overall institutional holdings increased by 1.4%, showing continued mutual fund and pension inflows."
    },
    "TSLA": {
        "activity": "Net Insider Buying",
        "institutional_change_pct": -0.5,
        "recent_trades": [
            {"date": "2026-06-02", "insider": "Musk Elon (CEO)", "type": "Award/Acquisition", "shares": 5000000, "price": 0.0},
            {"date": "2026-04-18", "insider": "Taneja Vaibhav (CFO)", "type": "Acquisition (Option Exercise)", "shares": 15000, "price": 142.10}
        ],
        "summary": "No major open-market insider sales recorded recently. Institutional holdings remain stable, though some hedge funds trimmed exposure to reallocate into chips."
    }
}

def analyze(ticker: str) -> Dict[str, Any]:
    """
    Insider Trading Agent
    Tracks CEO buying, director selling, institutional allocation shifts, and block trades.
    """
    ticker_upper = ticker.upper()
    
    # Get base data
    base_data = POPULAR_INSIDER_DATA.get(ticker_upper)
    
    if not base_data:
        base_data = {
            "activity": "Neutral / Quiet",
            "institutional_change_pct": 0.2,
            "recent_trades": [
                {"date": "2026-05-24", "insider": "Officer / Director", "type": "Sale (Option Exercise)", "shares": 5000, "price": 100.00}
            ],
            "summary": f"Insider activity for {ticker_upper} remains quiet, with only small, scheduled sales for executive compensation purposes. Institutional inflows are stable."
        }
        
    prompt = f"""
    You are the Insider Trading Agent. Analyze the insider transactions and institutional movements for {ticker}:
    - Activity Trend: {base_data['activity']}
    - Institutional Holding Change: {base_data['institutional_change_pct']}%
    - Recent Trades: {json.dumps(base_data['recent_trades'])}
    - Summary Baseline: {base_data['summary']}
    
    Write a formal report detailing insider alignment, block trades significance, and mutual fund accumulation trends.
    Format your response as a JSON object with:
    "activity": "description of activity posture",
    "institutional_change_pct": float representing percentage change,
    "recent_trades": list of trade dicts,
    "summary": "comprehensive narrative paragraph summarizing the insider alignment"
    """
    
    llm_output = generate_text(prompt, system_instruction="You are a financial investigator and institutional proxy advisor.")
    
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
            print(f"Failed to parse Insider Agent LLM output: {e}")
            
    return base_data
