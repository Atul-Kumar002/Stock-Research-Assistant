import json
from typing import Dict, Any
from .llm_helper import generate_text

def analyze(ticker: str, risk_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quantitative Risk Agent
    Evaluates volatility, Sharpe, Sortino, Max Drawdown, VaR, and Monte Carlo simulations.
    """
    if "error" in risk_data:
        return {
            "risk_level": "Medium",
            "explanation": "Risk data unavailable.",
            "metrics": risk_data,
            "monte_carlo_summary": {}
        }
        
    vol = risk_data["volatility"]
    sharpe = risk_data["sharpe"]
    sortino = risk_data["sortino"]
    max_dd = risk_data["max_drawdown"]
    beta = risk_data["beta"]
    var_95 = risk_data["var_95"]
    
    # Heuristics for risk level
    if vol > 0.35 or beta > 1.4 or max_dd < -0.35:
        risk_level = "High"
    elif vol < 0.18 and beta < 0.9 and max_dd > -0.15:
        risk_level = "Low"
    else:
        risk_level = "Medium"
        
    prompt = f"""
    You are the Quantitative Risk Agent. Evaluate the risk profile of {ticker} with these parameters:
    - Annualized Volatility: {vol * 100:.1f}%
    - Sharpe Ratio: {sharpe:.2f}
    - Sortino Ratio: {sortino:.2f}
    - Maximum Drawdown: {max_dd * 100:.1f}%
    - Beta: {beta:.2f}
    - 95% 1-day Value at Risk (VaR): {var_95 * 100:.2f}%
    
    Provide an institutional-grade quantitative risk assessment. Explain what these risk ratios mean for a portfolio, what maximum historical drawdowns tell us, and characterize the risk level (High/Medium/Low).
    Format your response as a JSON object with:
    "risk_level": "High/Medium/Low",
    "explanation": "narrative paragraph summarizing risk evaluation, referencing volatility, beta, and Sharpe ratios."
    """
    
    llm_output = generate_text(prompt, system_instruction="You are a financial risk manager and quantitative portfolio designer.")
    
    if llm_output:
        try:
            clean_output = llm_output.strip()
            if clean_output.startswith("```json"):
                clean_output = clean_output[7:]
            if clean_output.endswith("```"):
                clean_output = clean_output[:-3]
            data = json.loads(clean_output.strip())
            # Inject computed details
            data["metrics"] = {
                "volatility": vol,
                "sharpe": sharpe,
                "sortino": sortino,
                "max_drawdown": max_dd,
                "beta": beta,
                "var_95": var_95
            }
            data["monte_carlo"] = risk_data.get("monte_carlo", {})
            return data
        except Exception as e:
            print(f"Failed to parse Risk Agent LLM output: {e}")
            
    # Fallback simulation
    risk_descriptions = {
        "High": "characterized by elevated volatility and high beta, making it suitable only for aggressive expansion accounts.",
        "Medium": "presenting balanced volatility and beta indicators. It carries typical equity market risk with acceptable drawdown metrics.",
        "Low": "representing stable cash-generation profiles with low price fluctuations and defensive asset characteristics."
    }
    
    return {
        "risk_level": risk_level,
        "explanation": f"The quantitative risk profile for {ticker} is graded as {risk_level}, {risk_descriptions[risk_level]} It displays a volatility of {vol * 100:.1f}% and a Sharpe ratio of {sharpe:.2f}. The maximum historical drawdown peak-to-trough is {max_dd * 100:.1f}%, and the portfolio Beta of {beta:.2f} indicates corresponding market sensitivity.",
        "metrics": {
            "volatility": vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "beta": beta,
            "var_95": var_95
        },
        "monte_carlo": risk_data.get("monte_carlo", {})
    }
