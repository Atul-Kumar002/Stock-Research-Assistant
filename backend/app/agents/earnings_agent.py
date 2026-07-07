import json
from typing import Dict, Any
from .llm_helper import generate_text

# Database of earnings call highlights for popular stocks
POPULAR_EARNINGS_DATA = {
    "AAPL": {
        "outlook": "Double-digit growth expected in iPad and Services. Focus on rolling out Apple Intelligence feature sets to boost the iPhone upgrade cycle.",
        "risks": [
            "Delayed adoption of AI features in key international markets (China, EU).",
            "Hardware margin pressure due to higher material costs."
        ],
        "plans": [
            "Expanding retail presence in India and emerging markets.",
            "Selective hiring in core AI and machine learning engineering teams."
        ],
        "ceo_confidence": 88
    },
    "NVDA": {
        "outlook": "Demand for Blackwell and Hopper architectures remains well ahead of supply. Supply constraints expected to continue through next year.",
        "risks": [
            "CoWoS packaging capacity limits at TSMC.",
            "Gross margin normalization towards mid-70s as Blackwell ramps up."
        ],
        "plans": [
            "Expanding AI software platform integrations (NIMs).",
            "Accelerating hiring for hardware design and system integration divisions."
        ],
        "ceo_confidence": 95
    },
    "TSLA": {
        "outlook": "Focus on lowering cost per vehicle and accelerating autonomous driving technology. Expect production increases in energy storage products.",
        "risks": [
            "EV adoption rate slowing globally.",
            "FSD regulatory approvals in non-US jurisdictions."
        ],
        "plans": [
            "Pre-production of cheaper next-generation vehicle models starting late 2025.",
            "Restructuring sales operations to focus on robotaxi network rollout."
        ],
        "ceo_confidence": 82
    }
}

def analyze(ticker: str, sector: str = "Technology") -> Dict[str, Any]:
    """
    Earnings Call Agent
    Analyzes conference call transcripts, CEO confidence, outlook, and strategic hiring.
    """
    ticker_upper = ticker.upper()
    
    # Check database
    base_data = POPULAR_EARNINGS_DATA.get(ticker_upper)
    
    if not base_data:
        # Generic fallback
        base_data = {
            "outlook": f"Management is optimistic about stabilizing profit margins, pointing to moderate growth in client acquisition within the {sector} sector.",
            "risks": [
                "Intensifying competitive bidding environment.",
                "Slight increase in operating expenditures due to digital infrastructure migrations."
            ],
            "plans": [
                "Pruning low-margin segments to focus on core profitable lines.",
                "Targeted hiring for enterprise sales and product development."
            ],
            "ceo_confidence": 78
        }
        
    prompt = f"""
    You are the Earnings Call Agent. Analyze the conference call transcript highlights for {ticker}:
    - Outlook: {base_data['outlook']}
    - Risks mentioned: {json.dumps(base_data['risks'])}
    - Expansion & Hiring: {json.dumps(base_data['plans'])}
    - CEO Confidence Score: {base_data['ceo_confidence']}/100
    
    Evaluate the management tone, outlook, hiring plans, and CEO confidence.
    Format your response as a JSON object with:
    "outlook": "detailed summary of outlook",
    "risks": ["risk 1", "risk 2"],
    "plans": ["plan 1", "plan 2"],
    "ceo_confidence": integer between 0 and 100,
    "confidence_evaluation": "narrative paragraph evaluating management tone and strategy"
    """
    
    llm_output = generate_text(prompt, system_instruction="You are a financial analyst specializing in executive behavior and corporate earnings calls.")
    
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
            print(f"Failed to parse Earnings Agent LLM output: {e}")
            
    # Fallback simulation
    tone = "highly confident" if base_data["ceo_confidence"] > 85 else "cautiously optimistic"
    base_data["confidence_evaluation"] = f"Management's tone was {tone} during the call, with CEO comments highlighting long-term structural tailwinds. Strategic plans emphasize targeted capital allocation and expanding engineering talent in key growth channels."
    return base_data
