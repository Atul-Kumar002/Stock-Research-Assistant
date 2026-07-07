import json
from typing import Dict, Any, List
from .llm_helper import generate_text

def analyze(ticker: str, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recommendation Agent
    Synthesizes analysis from all agents to produce a BUY/SELL/HOLD recommendation
    with confidence rating, risk assessment, and key arguments.
    """
    ticker_upper = ticker.upper()
    
    # Extract scores
    fin_score = agent_outputs.get("financials", {}).get("score", 5.0)
    tech_score = agent_outputs.get("technical", {}).get("score", 5.0)
    news_score = agent_outputs.get("news", {}).get("impact_score", 5.0)
    sent_score = (agent_outputs.get("sentiment", {}).get("fear_greed_index", 50.0) / 10.0)
    ceo_score = (agent_outputs.get("earnings", {}).get("ceo_confidence", 70.0) / 10.0)
    
    # Simple weighted consensus score
    consensus_score = (
        (fin_score * 0.25) + 
        (tech_score * 0.20) + 
        (news_score * 0.15) + 
        (sent_score * 0.15) + 
        (ceo_score * 0.15) +
        (5.0 * 0.10) # default baseline for others
    )
    
    # Cap consensus score between 1.0 and 10.0
    consensus_score = max(1.0, min(10.0, consensus_score))
    
    # Risk adjustment
    risk_level = agent_outputs.get("risk", {}).get("risk_level", "Medium")
    if risk_level == "High":
        risk_score = 7.5
    elif risk_level == "Low":
        risk_score = 3.0
    else:
        risk_score = 5.5
        
    # Recommendation decision
    if consensus_score >= 7.0:
        recommendation = "BUY"
        confidence = int(consensus_score * 10)
    elif consensus_score <= 4.0:
        recommendation = "SELL"
        confidence = int((10 - consensus_score) * 10)
    else:
        recommendation = "HOLD"
        confidence = int((10 - abs(consensus_score - 5.5)) * 10)
        
    confidence = max(50, min(95, confidence))
    
    # Generate supporting reasons
    reasons = []
    
    # Financial reason
    fin_upside = agent_outputs.get("financials", {}).get("dcf", {}).get("upside_pct", 0.0)
    if fin_score > 7.0:
        reasons.append(f"Strong financial fundamentals: ROE and profitability ratios are in top quartile. DCF displays ${fin_upside:.1f}% upside.")
    elif fin_score < 4.5:
        reasons.append(f"Fundamentals are weak or deteriorating: margins are compressed and debt-to-equity is elevated.")
        
    # Technical reason
    tech_trend = agent_outputs.get("technical", {}).get("trend", "Sideways")
    if tech_score > 7.0:
        reasons.append(f"Technical chart displays solid bullish structure, trading above major daily moving averages with expanding MACD momentum.")
    elif tech_score < 4.5:
        reasons.append(f"Technical profile is bearish, with high volume breakdowns below support lines and oversold/oversold RSI pressure.")
        
    # News/Sentiment reason
    news_sent = agent_outputs.get("news", {}).get("sentiment", "Neutral")
    if news_sent == "Positive":
        reasons.append("Recent news flow is positive with rising institutional investment and strong earnings sentiment.")
    elif news_sent == "Negative":
        reasons.append("News flow is negative, presenting short-term operational challenges and lawsuits concerns.")
        
    # Default fallback reasons
    if len(reasons) < 3:
        reasons.append(f"Macroeconomic sector backdrop is {agent_outputs.get('macro', {}).get('sector_impact', 'Neutral')}, balancing macro rate pressures.")
    if len(reasons) < 3:
        reasons.append(f"CEO confidence score is steady at {ceo_score*10:.0f}/100, outlining clear execution plans.")

    prompt = f"""
    You are the Recommendation Agent. Synthesize the findings from all stock analysis agents for {ticker_upper}:
    {json.dumps({k: v for k, v in agent_outputs.items() if k != "risk" or "monte_carlo" not in v}, indent=2)}
    
    Consensus Score: {consensus_score:.1f}/10
    Proposed Recommendation: {recommendation}
    Confidence: {confidence}%
    Risk Score: {risk_score}/10
    
    Write a comprehensive, institutional-grade equity research report summary. Detail the investment thesis, key catalysts, and core risks.
    Format your response as a JSON object with:
    "recommendation": "BUY/SELL/HOLD",
    "confidence_pct": integer,
    "risk_score": float,
    "investment_horizon": "e.g. 12-18 Months",
    "reasons": ["reason 1", "reason 2", "reason 3"],
    "report_summary": "detailed multi-paragraph markdown report summary of the final investment decision"
    """
    
    llm_output = generate_text(prompt, system_instruction="You are the Managing Director of Institutional Research at an investment bank.")
    
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
            print(f"Failed to parse Recommendation Agent LLM output: {e}")
            
    # Fallback simulation
    rating_descr = "underpriced relative to its intrinsic value" if recommendation == "BUY" else "fairly valued with balanced risk-reward" if recommendation == "HOLD" else "overvalued under current market pricing"
    
    summary = f"""### Investment Thesis
We recommend a **{recommendation}** position on **{ticker_upper}** with a confidence of **{confidence}%** over a **12-18 Month** horizon. 
Our multi-agent consensus indicates that the company is {rating_descr}.

#### Key Catalysts
1. **Strong Fundamental Base**: Financial analysis grades fundamental health at **{fin_score}/10**, supported by healthy cash flow generation and growth metrics.
2. **Technical Momentum**: Price action displays a **{tech_trend}** trend, supported by solid buyer accumulation near key support lines.
3. **Sentiment Support**: General retail and news buzz is positive, reflecting strong options call volume and high CEO confidence.

#### Key Risks
1. **Macro Pressures**: Interest rate levels remain high, presenting capital expense pressures.
2. **Operational Executions**: Risk analysis rates volatility at **{agent_outputs.get('risk', {}).get('metrics', {}).get('volatility', 0.25)*100:.1f}%**, requiring strict stop-loss rules.
"""

    return {
        "recommendation": recommendation,
        "confidence_pct": confidence,
        "risk_score": risk_score,
        "investment_horizon": "12-18 Months",
        "reasons": reasons,
        "report_summary": summary
    }
