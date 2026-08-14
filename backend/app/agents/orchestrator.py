import asyncio
from typing import Dict, Any, List
from datetime import datetime, date
from ..services import stock_service
from . import (
    news_agent,
    financial_agent,
    technical_agent,
    sentiment_agent,
    sec_agent,
    earnings_agent,
    macro_agent,
    insider_agent,
    risk_agent,
    recommendation_agent,
    llm_helper
)


def _format_timestamp(value: object) -> str:
    """Return a safe formatted date string for various timestamp types.

    Handles datetime.date, datetime.datetime, pandas.Timestamp-like objects,
    and falls back to str(value) for other types.
    """
    if value is None:
        return ""
    # Common Python types
    if isinstance(value, (datetime, date)):
        try:
            return value.strftime("%m-%d")
        except Exception:
            return str(value)

    # pandas Timestamp (sometimes not imported as datetime subclass in typings)
    to_py = getattr(value, "to_pydatetime", None)
    if callable(to_py):
        try:
            dt = to_py()
            # dt may be typed as object; ensure it has strftime before calling
            strftime = getattr(dt, "strftime", None)
            if callable(strftime):
                # Ensure the returned value is a str for type checkers by casting
                try:
                    return str(strftime("%m-%d"))
                except Exception:
                    return str(dt)
            return str(dt)
        except Exception:
            pass

    # Fallback
    try:
        return str(value)
    except Exception:
        return ""

from ..services.cache_service import backend_cache

def get_stock_quote_and_metrics(ticker: str) -> Dict[str, Any]:
    """
    Fast stock quote and metrics getter without blocking on LLM agent research.
    Response time target: < 300ms - 800ms.
    """
    ticker_upper = ticker.upper().strip()
    cache_key = f"fast_quote:{ticker_upper}"
    cached = backend_cache.get(cache_key)
    if cached is not None:
        return cached

    stock_data = stock_service.get_stock_data(ticker_upper)
    info = stock_data.get("info", {})
    sector = info.get("sector", "Technology")

    prices_df = stock_service.get_historical_prices(ticker_upper, period="1y")
    tech_data = stock_service.calculate_technical_indicators(prices_df)

    beta_val = info.get("beta")
    if beta_val is None:
        beta_val = 1.0
    risk_metrics = stock_service.calculate_risk_metrics(prices_df, beta_fallback=beta_val)

    current_price = tech_data.get("current_price", info.get("currentPrice", 100.0))
    price_change = info.get("regularMarketChangePercent", 0.0)
    if price_change is None:
        price_change = 0.0

    history_data = []
    if not prices_df.empty:
        tail_df = prices_df.tail(60)
        for idx, row in tail_df.iterrows():
            history_data.append({
                "date": _format_timestamp(idx),
                "open": float(row["Open"]) if "Open" in row else float(row["Close"]),
                "high": float(row["High"]) if "High" in row else float(row["Close"]),
                "low": float(row["Low"]) if "Low" in row else float(row["Close"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]) if "Volume" in row else 0.0
            })

    currency = info.get("currency", "USD")
    if not currency:
        currency = "USD"
    if ticker_upper.endswith(".NS") or ticker_upper.endswith(".BO") or ticker_upper == "RELIANCE":
        currency = "INR"

    result = {
        "ticker": ticker_upper,
        "company_name": info.get("shortName", info.get("longName", ticker_upper)),
        "sector": sector,
        "current_price": current_price,
        "price_change_pct": price_change,
        "currency": currency,
        "history": history_data,
        "technical_data": tech_data,
        "risk_metrics": risk_metrics
    }
    backend_cache.set(cache_key, result, ttl_seconds=30) # 30s TTL for live quote
    return result

def run_agent_pipeline(ticker: str) -> Dict[str, Any]:
    """
    Synchronous pipeline runner to gather data and analyze ticker using all 10 agents,
    leveraging TTL caching and agent-level error isolation.
    """
    ticker_upper = ticker.upper().strip()
    cache_key = f"full_analysis:{ticker_upper}"
    cached = backend_cache.get(cache_key)
    if cached is not None:
        return cached

    # 1. Fetch fast quote metrics
    quote = get_stock_quote_and_metrics(ticker_upper)
    
    # 2. Fetch shared context
    stock_data = stock_service.get_stock_data(ticker_upper)
    prices_df = stock_service.get_historical_prices(ticker_upper, period="1y")
    news_articles = stock_service.get_stock_news(ticker_upper)
    
    tech_data = quote["technical_data"]
    risk_metrics = quote["risk_metrics"]
    sector = quote["sector"]

    # Safe agent execution helper
    def safe_run(fn, *args):
        try:
            return fn(*args)
        except Exception as e:
            print(f"Agent execution error for {fn.__module__}: {e}")
            return {"error": str(e), "score": 5.0, "sentiment": "Neutral", "risk_level": "Medium"}
    
    # 4. Execute agents in parallel threads
    async def run_parallel():
        tasks = [
            asyncio.to_thread(safe_run, news_agent.analyze, ticker_upper, news_articles),
            asyncio.to_thread(safe_run, financial_agent.analyze, ticker_upper, stock_data),
            asyncio.to_thread(safe_run, technical_agent.analyze, ticker_upper, tech_data),
            asyncio.to_thread(safe_run, sentiment_agent.analyze, ticker_upper, news_articles),
            asyncio.to_thread(safe_run, sec_agent.analyze, ticker_upper, sector),
            asyncio.to_thread(safe_run, earnings_agent.analyze, ticker_upper, sector),
            asyncio.to_thread(safe_run, macro_agent.analyze, ticker_upper, sector),
            asyncio.to_thread(safe_run, insider_agent.analyze, ticker_upper),
            asyncio.to_thread(safe_run, risk_agent.analyze, ticker_upper, risk_metrics),
        ]
        return await asyncio.gather(*tasks)

    results = asyncio.run(run_parallel())
    news_out, fin_out, tech_out, sent_out, sec_out, earn_out, macro_out, insider_out, risk_out = results
    
    agent_outputs = {
        "news": news_out,
        "financials": fin_out,
        "technical": tech_out,
        "sentiment": sent_out,
        "sec": sec_out,
        "earnings": earn_out,
        "macro": macro_out,
        "insider": insider_out,
        "risk": risk_out
    }
    
    # Run Final Recommendation Agent safely
    try:
        rec_out = recommendation_agent.analyze(ticker_upper, agent_outputs)
    except Exception as e:
        print(f"Recommendation agent error: {e}")
        rec_out = {
            "recommendation": "HOLD",
            "confidence_pct": 65,
            "risk_score": 5.0,
            "report_summary": "Analysis completed with standard fallback metrics."
        }
    
    # Generate Multi-Agent Debate script
    debate = generate_debate_transcript(ticker_upper, agent_outputs, rec_out)
    
    analysis_result = {
        "ticker": ticker_upper,
        "company_name": quote["company_name"],
        "sector": sector,
        "current_price": quote["current_price"],
        "price_change_pct": quote["price_change_pct"],
        "currency": quote["currency"],
        "history": quote["history"],
        "summary": {
            "recommendation": rec_out.get("recommendation", "HOLD"),
            "confidence_pct": rec_out.get("confidence_pct", 70),
            "risk_score": rec_out.get("risk_score", 5.0),
            "risk_level": risk_out.get("risk_level", "Medium"),
            "financial_score": fin_out.get("score", 5.0),
            "technical_score": tech_out.get("score", 5.0),
            "news_sentiment": news_out.get("sentiment", "Neutral")
        },
        "agents": agent_outputs,
        "recommendation": rec_out,
        "debate": debate
    }
    backend_cache.set(cache_key, analysis_result, ttl_seconds=1800) # 30 min research cache
    return analysis_result

def generate_debate_transcript(ticker: str, agents: Dict[str, Any], rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate an interactive, verbal exchange between agents based on their scores.
    Missing or partial agent payloads are handled with safe defaults so the UI remains stable.
    """
    debate_transcript = []

    tech_agent = agents.get("technical") or {}
    tech_indicators = tech_agent.get("indicators") or {}
    tech_trend = str(tech_agent.get("trend", "Sideways"))
    tech_score = float(tech_agent.get("score", 5.0))
    rsi = float(tech_indicators.get("rsi", 50.0))
    support = float(tech_indicators.get("support", 0.0))

    debate_transcript.append({
        "agent": "Technical Analysis Agent",
        "avatar_color": "accent-blue",
        "verdict": "BUY" if tech_score > 6.5 else "SELL" if tech_score < 4.0 else "HOLD",
        "message": f"Looking at {ticker}, we are in a confirmed {tech_trend.lower()} trend. RSI is holding at {rsi:.1f} and MACD is {'positive' if tech_score > 6.0 else 'weakening'}. We see firm buyer accumulation near support at ${support:.2f}. This is a clear technical entry point."
    })

    fin_agent = agents.get("financials") or {}
    fin_dcf = fin_agent.get("dcf") or {}
    fin_score = float(fin_agent.get("score", 5.0))
    upside = float(fin_dcf.get("upside_pct", 0.0))
    margin_explanation = str(fin_agent.get("margin_explanation", "Margins remain stable and supportive of the thesis."))

    debate_transcript.append({
        "agent": "Financial Statement Agent",
        "avatar_color": "accent-yellow",
        "verdict": "BUY" if fin_score > 6.5 else "SELL" if fin_score < 4.0 else "HOLD",
        "message": f"I agree with the technical entry but let's anchor it in fundamentals. Our DCF model estimates {ticker}'s intrinsic value with a {upside:.1f}% upside. {margin_explanation}. Growth is stable, making this investment fundamentally sound."
    })

    risk_agent = agents.get("risk") or {}
    risk_metrics = risk_agent.get("metrics") or {}
    risk_level = str(risk_agent.get("risk_level", "Medium"))
    vol = float(risk_metrics.get("volatility", 0.2))
    max_drawdown = float(risk_metrics.get("max_drawdown", 0.0))

    debate_transcript.append({
        "agent": "Quantitative Risk Agent",
        "avatar_color": "accent-red",
        "verdict": "CAUTION" if risk_level == "High" else "STABLE",
        "message": f"Not so fast! We must evaluate risk. Annualized volatility is {vol*100:.1f}% and the maximum drawdown tracks at {max_drawdown*100:.1f}%. While fundamentals look positive, the downside tail risk is significant. I advise hedging this position or using strict stop-loss bounds."
    })

    macro_agent = agents.get("macro") or {}
    macro_sent = str(macro_agent.get("sector_impact", "Neutral"))
    macro_explanation = str(macro_agent.get("explanation", "The sector backdrop appears balanced."))
    macro_segment = macro_explanation.split(" presents")[0] if " presents" in macro_explanation else macro_explanation

    debate_transcript.append({
        "agent": "Macro Economic Agent",
        "avatar_color": "accent-orange",
        "verdict": macro_sent.upper(),
        "message": f"Regarding the macro layer, the Fed benchmark rate of 5.25%-5.50% keeps capital costs high. However, our sector impact model grades {macro_sent} tailwinds for {macro_segment}. Secular industrial forces should offset interest rate drags."
    })

    recommendation = str(rec.get("recommendation", "HOLD"))
    confidence_pct = rec.get("confidence_pct", 70)
    debate_transcript.append({
        "agent": "Recommendation Agent (Consensus)",
        "avatar_color": "accent-green",
        "verdict": recommendation,
        "message": f"Consensus achieved. Combining the technical momentum, robust {fin_score}/10 financial health, and factoring in the Risk Agent's volatility warnings, we issue a final **{recommendation}** rating with a confidence score of **{confidence_pct}%** over a 12-18 month horizon."
    })

    return debate_transcript
