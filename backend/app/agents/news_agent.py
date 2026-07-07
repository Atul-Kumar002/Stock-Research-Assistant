import json
from typing import Dict, Any, List
from .llm_helper import generate_text

def analyze(ticker: str, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    News Intelligence Agent
    Analyzes news articles, filters duplicates, rates impact, and generates summaries.
    """
    if not news_articles:
        return {
            "summary": "No recent news found for this ticker.",
            "importance": "Low",
            "confidence_pct": 50,
            "impact_score": 5.0,
            "sentiment": "Neutral",
            "articles": []
        }

    # Filter out exact duplicates based on title
    seen_titles = set()
    filtered_articles = []
    for art in news_articles:
        t = art["title"].strip()
        if t not in seen_titles:
            seen_titles.add(t)
            filtered_articles.append(art)

    # Calculate average impact score from article heuristic scores
    scores = [art["score"] for art in filtered_articles]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    # Map score to sentiment
    if avg_score > 0.15:
        sentiment = "Positive"
        impact_score = 5.0 + (avg_score * 5.0) # 5 to 10
    elif avg_score < -0.15:
        sentiment = "Negative"
        impact_score = 5.0 + (avg_score * 5.0) # 0 to 5
    else:
        sentiment = "Neutral"
        impact_score = 5.0
        
    impact_score = max(0.0, min(10.0, impact_score))
    
    # Set importance based on score absolute value and article count
    abs_score = abs(avg_score)
    if abs_score > 0.4 or len(filtered_articles) > 5:
        importance = "High"
    elif abs_score > 0.2:
        importance = "Medium"
    else:
        importance = "Low"

    # Try to use LLM for summary and intelligence
    prompt = f"""
    You are the News Intelligence Agent. Analyze the following news articles for {ticker}:
    {json.dumps(filtered_articles, indent=2)}
    
    Provide a professional, concise summary of the key news themes, duplicates filtered, importance, impact score (0-10), confidence %, and overall sentiment (Positive, Negative, Neutral).
    Format your response as a JSON object with:
    "summary": "a short narrative summarizing the news flow",
    "importance": "High/Medium/Low",
    "confidence_pct": integer between 0 and 100,
    "impact_score": float between 0.0 and 10.0,
    "sentiment": "Positive/Negative/Neutral"
    """
    
    llm_output = generate_text(prompt, system_instruction="You are an institutional financial analyst.")
    
    if llm_output:
        try:
            # Try to strip markdown code blocks if any
            clean_output = llm_output.strip()
            if clean_output.startswith("```json"):
                clean_output = clean_output[7:]
            if clean_output.endswith("```"):
                clean_output = clean_output[:-3]
            data = json.loads(clean_output.strip())
            # Inject filtered articles
            data["articles"] = filtered_articles
            return data
        except Exception as e:
            print(f"Failed to parse News Agent LLM output: {e}")
            
    # Fallback simulation
    pos_count = sum(1 for a in filtered_articles if a["sentiment"] == "Positive")
    neg_count = sum(1 for a in filtered_articles if a["sentiment"] == "Negative")
    
    if pos_count > neg_count:
        summary_text = f"News flow is positive, led by recent highlights regarding growth initiatives and strong market presence. We filtered {len(news_articles) - len(filtered_articles)} duplicates."
    elif neg_count > pos_count:
        summary_text = f"News flow shows bearish pressure, with articles highlighting external market concerns, high interest rates, or supply chain constraints. We filtered {len(news_articles) - len(filtered_articles)} duplicates."
    else:
        summary_text = f"News flow is mixed or quiet, with standard corporate updates and sideways sentiment. We filtered {len(news_articles) - len(filtered_articles)} duplicates."
        
    return {
        "summary": summary_text,
        "importance": importance,
        "confidence_pct": 80,
        "impact_score": round(impact_score, 1),
        "sentiment": sentiment,
        "articles": filtered_articles
    }
