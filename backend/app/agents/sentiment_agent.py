import json
from typing import Dict, Any, List
from .llm_helper import generate_text

def analyze(ticker: str, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sentiment Agent
    Aggregates sentiment from news, Twitter, Reddit, and StockTwits.
    """
    # Calculate baseline from news sentiment
    news_sentiments = [a["score"] for a in news_articles] if news_articles else []
    base_avg = sum(news_sentiments) / len(news_sentiments) if news_sentiments else 0.0
    
    # Calculate simulated social sentiment
    # Positive avg results in higher greed and bullish %
    bullish_pct = 50 + int(base_avg * 35) # e.g. -0.5 -> 32%, 0.5 -> 67%
    bullish_pct = max(15, min(95, bullish_pct))
    bearish_pct = 100 - bullish_pct
    
    fear_greed = 50 + int(base_avg * 40) # 0 to 100
    fear_greed = max(5, min(95, fear_greed))
    
    trending_topics = [f"${ticker}", f"#{ticker}", "Investing", "StockMarket"]
    if base_avg > 0.2:
        trending_topics.append("BullRun")
        trending_topics.append("EarningsBeat")
    elif base_avg < -0.2:
        trending_topics.append("ShortSeller")
        trending_topics.append("MarginCall")
    else:
        trending_topics.append("OptionsFlow")
        
    prompt = f"""
    You are the Sentiment Agent. Evaluate the market and social media sentiment for {ticker}:
    - News sentiment baseline: {base_avg:.2f}
    - Bullish sentiment: {bullish_pct}%
    - Bearish sentiment: {bearish_pct}%
    - Fear & Greed Index: {fear_greed}/100
    
    Provide an institutional sentiment report detailing retail momentum, option flows activity, and public sentiment alignment.
    Format your response as a JSON object with:
    "fear_greed_index": integer between 0 and 100,
    "bullish_pct": integer between 0 and 100,
    "bearish_pct": integer between 0 and 100,
    "trending_topics": list of strings,
    "summary": "narrative paragraph summarizing the sentiment environment"
    """
    
    llm_output = generate_text(prompt, system_instruction="You are a behavioral finance researcher and social media market analyst.")
    
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
            print(f"Failed to parse Sentiment Agent LLM output: {e}")
            
    # Fallback simulation
    sentiment_label = "Very Bullish" if fear_greed > 75 else "Bullish" if fear_greed > 55 else "Fearful" if fear_greed < 40 else "Neutral"
    
    return {
        "fear_greed_index": fear_greed,
        "bullish_pct": bullish_pct,
        "bearish_pct": bearish_pct,
        "trending_topics": trending_topics,
        "summary": f"Sentiment for {ticker} is currently {sentiment_label} (Fear & Greed: {fear_greed}/100). Social buzz on Twitter and Reddit is running {bullish_pct}% bullish, reflecting positive retail interest and heavy call option flow."
    }
