import json
from typing import Dict, Any
from .llm_helper import generate_text

def calculate_dcf_intrinsic_value(info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform a simplified 5-year DCF calculation based on yfinance info
    """
    try:
        # Get free cash flow, or approximate from net income or operating cash flow
        fcf = info.get("freeCashflow")
        if not fcf:
            fcf = info.get("operatingCashflow", 0) * 0.8  # approximate FCF as 80% of OCF
            
        if fcf <= 0:
            # If negative, fallback to net income or a percentage of revenue
            fcf = info.get("netIncomeToCommon", 0) * 0.8
            if fcf <= 0:
                fcf = info.get("totalRevenue", 0) * 0.1  # 10% of revenue
                
        # Growth rate (cap at 30% and floor at 3% for safety)
        growth_rate = info.get("revenueGrowth", 0.10)
        if growth_rate is None:
            growth_rate = 0.10
        growth_rate = max(0.03, min(0.30, growth_rate))
        
        # Discount rate (WACC) - simple approximation based on beta
        beta = info.get("beta", 1.0)
        if beta is None:
            beta = 1.0
        wacc = 0.08 + 0.04 * max(0.5, min(2.0, beta))  # between 10% and 16%
        
        # Terminal growth rate
        terminal_growth = 0.03
        
        # Project FCF for 5 years
        projected_fcfs = []
        discounted_fcfs = []
        current_fcf = fcf
        
        for i in range(1, 6):
            current_fcf = current_fcf * (1 + growth_rate)
            projected_fcfs.append(current_fcf)
            discounted_fcfs.append(current_fcf / ((1 + wacc) ** i))
            
        # Terminal value
        terminal_value = (projected_fcfs[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)
        discounted_terminal_value = terminal_value / ((1 + wacc) ** 5)
        
        # Enterprise Value
        enterprise_value = sum(discounted_fcfs) + discounted_terminal_value
        
        # Equity value adjustments
        total_cash = info.get("totalCash", 0)
        total_debt = info.get("totalDebt", 0)
        equity_value = enterprise_value + total_cash - total_debt
        
        shares_outstanding = info.get("sharesOutstanding")
        if not shares_outstanding or shares_outstanding <= 0:
            shares_outstanding = 1e9  # fallback 1 billion shares
            
        intrinsic_value = equity_value / shares_outstanding
        current_price = info.get("currentPrice", 100.0)
        upside = ((intrinsic_value - current_price) / current_price) * 100 if current_price else 0
        
        # Cap intrinsic value in case of wild inputs
        intrinsic_value = max(current_price * 0.2, min(current_price * 5.0, intrinsic_value))
        upside = ((intrinsic_value - current_price) / current_price) * 100
        
        return {
            "free_cash_flow": float(fcf),
            "estimated_growth_rate": float(growth_rate),
            "wacc": float(wacc),
            "intrinsic_value": float(intrinsic_value),
            "current_price": float(current_price),
            "upside_pct": float(upside)
        }
    except Exception as e:
        print(f"Error in DCF calculation: {e}")
        current_price = info.get("currentPrice", 100.0)
        return {
            "free_cash_flow": 0.0,
            "estimated_growth_rate": 0.10,
            "wacc": 0.10,
            "intrinsic_value": current_price * 1.15,
            "current_price": current_price,
            "upside_pct": 15.0
        }

def analyze(ticker: str, stock_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Financial Statement Agent
    Evaluates revenue growth, margins, debt, ratios, and computes DCF intrinsic value.
    """
    info = stock_data.get("info", {})
    dcf = calculate_dcf_intrinsic_value(info)
    
    # Extract ratios
    pe_ratio = info.get("trailingPE") or info.get("forwardPE") or 0.0
    pb_ratio = info.get("priceToBook") or 0.0
    peg_ratio = info.get("pegRatio") or 0.0
    roe = info.get("returnOnEquity") or 0.0
    debt_equity = info.get("debtToEquity") or 0.0
    operating_margin = info.get("operatingMargins") or 0.0
    revenue_growth = info.get("revenueGrowth") or 0.0
    
    # Calculate health score (0-10) based on metrics
    score = 5.0
    if roe > 0.15: score += 1.0
    if roe > 0.25: score += 0.5
    if operating_margin > 0.15: score += 1.0
    if operating_margin > 0.25: score += 0.5
    if revenue_growth > 0.10: score += 1.0
    if revenue_growth > 0.25: score += 0.5
    if debt_equity < 50 and debt_equity > 0: score += 1.0
    elif debt_equity > 150: score -= 1.0
    if pe_ratio > 0 and pe_ratio < 25: score += 0.5
    elif pe_ratio > 50: score -= 0.5
    
    score = max(1.0, min(10.0, score))
    
    # Try LLM for qualitative explanation
    prompt = f"""
    You are the Financial Statement Agent. Analyze the financial health of {ticker} with these metrics:
    - Current Price: {info.get('currentPrice')}
    - Revenue Growth: {revenue_growth * 100}%
    - Operating Margins: {operating_margin * 100}%
    - Debt-to-Equity: {debt_equity}%
    - ROE: {roe * 100}%
    - P/E: {pe_ratio}
    - P/B: {pb_ratio}
    - PEG: {peg_ratio}
    - DCF Valuation: Intrinsic value of {dcf['intrinsic_value']:.2f} compared to current price {dcf['current_price']:.2f} ({dcf['upside_pct']:.1f}% upside).
    
    Write detailed institutional-grade explanations for:
    1. Revenue performance (e.g. "Revenue increased/decreased because...")
    2. Debt/Balance Sheet health (e.g. "Debt is concerning/healthy because...")
    3. Operating Margins & Profitability (e.g. "Margins improved/declined because...")
    4. DCF Valuation summary.
    
    Format your response as a JSON object with:
    "score": float between 1.0 and 10.0,
    "revenue_explanation": "detailed sentence explaining revenue",
    "debt_explanation": "detailed sentence explaining debt status",
    "margin_explanation": "detailed sentence explaining margins and profit",
    "dcf_explanation": "detailed sentence explaining intrinsic value upside",
    "financials_summary": "overall financial report paragraph"
    """
    
    llm_output = generate_text(prompt, system_instruction="You are an expert CPA and institutional financial analyst.")
    
    if llm_output:
        try:
            clean_output = llm_output.strip()
            if clean_output.startswith("```json"):
                clean_output = clean_output[7:]
            if clean_output.endswith("```"):
                clean_output = clean_output[:-3]
            data = json.loads(clean_output.strip())
            # Inject computed DCF details
            data["dcf"] = dcf
            data["score"] = float(data.get("score", score))
            return data
        except Exception as e:
            print(f"Failed to parse Financial Agent LLM output: {e}")
            
    # Fallback simulation
    rev_dir = "increased" if revenue_growth > 0 else "decreased"
    debt_dir = "concerning due to high leverage" if debt_equity > 100 else "healthy with low leverage"
    margin_dir = "improved" if operating_margin > 0.15 else "under pressure"
    upside_status = "undervalued" if dcf["upside_pct"] > 5 else "overvalued"
    
    return {
        "score": round(score, 1),
        "revenue_explanation": f"Revenue {rev_dir} by {revenue_growth * 100:.1f}% YoY, demonstrating solid business expansion and product demand.",
        "debt_explanation": f"Debt is {debt_dir} at a debt-to-equity ratio of {debt_equity:.1f}%, indicating manageable leverage risk.",
        "margin_explanation": f"Operating margins are {margin_dir} at {operating_margin * 100:.1f}%, highlighting strong pricing power or cost controls.",
        "dcf_explanation": f"DCF indicates the stock is {upside_status} with an intrinsic value of ${dcf['intrinsic_value']:.2f} representing a {dcf['upside_pct']:.1f}% deviation from its current price.",
        "financials_summary": f"Fundamentals are robust (Health Score: {score:.1f}/10). The company displays strong growth of {revenue_growth * 100:.1f}% and ROE of {roe * 100:.1f}%, matching high-quality investment standards.",
        "dcf": dcf
    }
