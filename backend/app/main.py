import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import uvicorn

from .database import init_db, get_db, Watchlist, SearchHistory, SavedPortfolio, ChatHistory
from .agents.orchestrator import run_agent_pipeline
from .agents.llm_helper import generate_text

# Initialize DB on startup
init_db()

app = FastAPI(title="Finance Assistant - Multi-Agent Backend", version="1.0.0")

# Enable CORS for the Astro frontend (typically runs on http://localhost:4321)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Popular stocks database for autocomplete and search
POPULAR_STOCKS = [
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", "price": 192.50, "currency": "USD"},
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology", "price": 125.20, "currency": "USD"},
    {"ticker": "TSLA", "name": "Tesla, Inc.", "sector": "Consumer Cyclical", "price": 178.90, "currency": "USD"},
    {"ticker": "RELIANCE", "name": "Reliance Industries Ltd.", "sector": "Energy/Conglomerate", "price": 2950.00, "currency": "INR"},
    {"ticker": "INFY", "name": "Infosys Limited", "sector": "Technology", "price": 18.50, "currency": "USD"},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "Technology", "price": 420.30, "currency": "USD"},
    {"ticker": "AMZN", "name": "Amazon.com, Inc.", "sector": "Consumer Cyclical", "price": 185.10, "currency": "USD"},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology", "price": 175.40, "currency": "USD"},
    {"ticker": "AMD", "name": "Advanced Micro Devices, Inc.", "sector": "Technology", "price": 160.80, "currency": "USD"}
]

# Request models
class PortfolioRequest(BaseModel):
    capital: float
    risk_appetite: str # Conservative, Moderate, Aggressive
    investment_horizon: int # years
    country: str = "US"

class ChatRequest(BaseModel):
    session_id: str
    message: str
    ticker: Optional[str] = None

@app.get("/api/exchange-rate")
def get_exchange_rate():
    """
    Fetch real-time USD to INR exchange rate from yfinance
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker("USDINR=X")
        history = ticker.history(period="1d")
        if not history.empty:
            rate = float(history['Close'].iloc[-1])
        else:
            rate = ticker.info.get('regularMarketPrice') or ticker.info.get('previousClose') or 83.50
        return {"rate": rate}
    except Exception as e:
        print(f"Error fetching exchange rate: {e}")
        return {"rate": 83.50}

@app.get("/api/search")
def search_stocks(q: str = Query("", min_length=0), db: Session = Depends(get_db)):
    """
    Search stocks with autocomplete support
    """
    q_clean = q.strip().upper()
    
    # Save search count in DB for analytics/popular list
    if q_clean:
        hist = db.query(SearchHistory).filter(SearchHistory.ticker == q_clean).first()
        if hist:
            hist.query_count += 1
        else:
            hist = SearchHistory(ticker=q_clean)
            db.add(hist)
        db.commit()
        
    if not q_clean:
        return POPULAR_STOCKS
        
    results = []
    for stock in POPULAR_STOCKS:
        if q_clean in stock["ticker"] or q_clean in stock["name"].upper():
            results.append(stock)
            
    # If no results in our popular list, return the ticker itself as a custom option
    if not results and len(q_clean) <= 6:
        currency = "INR" if (q_clean.endswith(".NS") or q_clean.endswith(".BO") or q_clean == "RELIANCE") else "USD"
        results.append({
            "ticker": q_clean,
            "name": f"{q_clean} Corporation",
            "sector": "General Sector",
            "price": 100.00,
            "currency": currency
        })
        
    return results

@app.get("/api/stock/{ticker}")
def analyze_stock(ticker: str, db: Session = Depends(get_db)):
    """
    Run multi-agent analysis for a specific ticker
    """
    ticker_clean = ticker.strip().upper()
    try:
        analysis_results = run_agent_pipeline(ticker_clean)
        return analysis_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run agent analysis pipeline: {str(e)}")

@app.get("/api/watchlist")
def get_watchlist(db: Session = Depends(get_db)):
    """
    Get all watchlisted tickers
    """
    items = db.query(Watchlist).all()
    # Populate with current names & prices
    enriched_items = []
    for item in items:
        # Find in popular or fallback
        match = next((s for s in POPULAR_STOCKS if s["ticker"] == item.ticker), None)
        if match:
            enriched_items.append({
                "ticker": item.ticker,
                "name": match["name"],
                "price": match["price"],
                "sector": match["sector"],
                "currency": match.get("currency", "USD")
            })
        else:
            currency = "INR" if (item.ticker.endswith(".NS") or item.ticker.endswith(".BO") or item.ticker == "RELIANCE") else "USD"
            enriched_items.append({
                "ticker": item.ticker,
                "name": f"{item.ticker} Corp",
                "price": 100.00,
                "sector": "General Sector",
                "currency": currency
            })
    return enriched_items

@app.post("/api/watchlist/{ticker}")
def add_to_watchlist(ticker: str, db: Session = Depends(get_db)):
    """
    Add a ticker to the watchlist
    """
    ticker_clean = ticker.strip().upper()
    existing = db.query(Watchlist).filter(Watchlist.ticker == ticker_clean).first()
    if existing:
        return {"message": f"{ticker_clean} already in watchlist."}
        
    item = Watchlist(ticker=ticker_clean)
    db.add(item)
    db.commit()
    return {"message": f"Added {ticker_clean} to watchlist."}

@app.delete("/api/watchlist/{ticker}")
def remove_from_watchlist(ticker: str, db: Session = Depends(get_db)):
    """
    Remove a ticker from the watchlist
    """
    ticker_clean = ticker.strip().upper()
    item = db.query(Watchlist).filter(Watchlist.ticker == ticker_clean).first()
    if not item:
        raise HTTPException(status_code=404, detail="Ticker not found in watchlist")
        
    db.delete(item)
    db.commit()
    return {"message": f"Removed {ticker_clean} from watchlist."}

@app.post("/api/portfolio")
def build_portfolio(req: PortfolioRequest, db: Session = Depends(get_db)):
    """
    Build a smart portfolio asset allocation
    """
    capital = req.capital
    risk = req.risk_appetite
    horizon = req.investment_horizon
    
    # Asset allocations based on risk profiles
    if risk == "Conservative":
        expected_return = 6.2
        beta = 0.45
        allocations = [
            {"asset": "Bonds & Treasuries", "weight": 50, "amount": capital * 0.50, "ticker": "TLT"},
            {"asset": "US Large Cap (Blue Chips)", "weight": 25, "amount": capital * 0.25, "ticker": "SPY"},
            {"asset": "Dividend Equities", "weight": 10, "amount": capital * 0.10, "ticker": "SCHD"},
            {"asset": "Gold & Commodities", "weight": 10, "amount": capital * 0.10, "ticker": "GLD"},
            {"asset": "Cash Cash Equivalents", "weight": 5, "amount": capital * 0.05, "ticker": "CASH"}
        ]
        explanation = "A conservative allocation focusing heavily on capital preservation and fixed-income assets. TLT (Long-Term Treasuries) and gold cushion equity volatility."
    elif risk == "Moderate":
        expected_return = 9.8
        beta = 0.85
        allocations = [
            {"asset": "US Large Cap Equities", "weight": 40, "amount": capital * 0.40, "ticker": "SPY"},
            {"asset": "Technology & Growth Growth", "weight": 20, "amount": capital * 0.20, "ticker": "QQQ"},
            {"asset": "Bonds & Fixed Income", "weight": 20, "amount": capital * 0.20, "ticker": "AGG"},
            {"asset": "International Stocks", "weight": 10, "amount": capital * 0.10, "ticker": "VXUS"},
            {"asset": "Gold & Commodities", "weight": 10, "amount": capital * 0.10, "ticker": "GLD"}
        ]
        explanation = "A balanced allocation matching index-growth with cash preservation. Focuses on broad-market ETFs (SPY, QQQ) combined with 20% defensive fixed income."
    else: # Aggressive
        expected_return = 14.5
        beta = 1.30
        allocations = [
            {"asset": "AI & Semiconductor Equities", "weight": 40, "amount": capital * 0.40, "ticker": "NVDA/AMD"},
            {"asset": "US Tech & Innovation Growth", "weight": 30, "amount": capital * 0.30, "ticker": "QQQ"},
            {"asset": "Emerging Tech / Disruption", "weight": 15, "amount": capital * 0.15, "ticker": "ARKK"},
            {"asset": "Large Cap Core", "weight": 10, "amount": capital * 0.10, "ticker": "AAPL"},
            {"asset": "Gold & Commodities", "weight": 5, "amount": capital * 0.05, "ticker": "GLD"}
        ]
        explanation = "An aggressive growth portfolio heavily weighted towards high-beta technology and artificial intelligence accelerators. High potential return but prone to significant drawdown."

    # Save portfolio in DB
    alloc_json = {item["asset"]: {"weight": item["weight"], "ticker": item["ticker"]} for item in allocations}
    saved_port = SavedPortfolio(capital=capital, risk_appetite=risk, horizon=horizon, allocation=alloc_json)
    db.add(saved_port)
    db.commit()
    
    return {
        "expected_return_pct": expected_return,
        "portfolio_beta": beta,
        "explanation": explanation,
        "allocations": allocations
    }

@app.post("/api/chat")
def chat_with_agent(req: ChatRequest, db: Session = Depends(get_db)):
    """
    RAG-grounded Stock Chatbot with question-aware fallback
    """
    session = req.session_id
    msg = req.message
    ticker = req.ticker

    # Save user message
    db.add(ChatHistory(session_id=session, role="user", content=msg))
    db.commit()

    # Fetch stock data for full RAG context
    stock_data = None
    rag_context = ""
    if ticker:
        try:
            stock_data = run_agent_pipeline(ticker)
            tech = stock_data["agents"]["technical"]
            ind = tech["indicators"]
            fin = stock_data["agents"]["financials"]
            news = stock_data["agents"]["news"]
            sent = stock_data["agents"]["sentiment"]
            risk = stock_data["agents"]["risk"]
            sec = stock_data["agents"]["sec"]
            earn = stock_data["agents"]["earnings"]
            summ = stock_data["summary"]

            rag_context = f"""
            === Finance Assistant RAG Context for {ticker} ===
            Company: {stock_data['company_name']} | Sector: {stock_data.get('sector', 'N/A')}
            Current Price: ${stock_data['current_price']:.2f} ({stock_data['price_change_pct']:+.2f}% today)

            [CONSENSUS]
            Recommendation: {summ['recommendation']} | Confidence: {summ['confidence_pct']}% | Risk Level: {summ['risk_level']} ({summ['risk_score']:.1f}/10)
            Financial Score: {summ['financial_score']:.1f}/10 | Technical Score: {summ['technical_score']:.1f}/10
            News Sentiment: {summ['news_sentiment']}

            [TECHNICAL ANALYSIS]
            Trend: {tech['trend']} | CMT Summary: {tech['summary']}
            RSI(14): {ind['rsi']:.1f} | MACD Histogram: {ind['macd_hist']:.4f}
            Support: ${ind['support']:.2f} | Resistance: ${ind['resistance']:.2f}
            SMA 20: ${ind['sma_20']:.2f} | SMA 50: ${ind['sma_50']:.2f}

            [FUNDAMENTAL / DCF]
            DCF Intrinsic Value: ${fin['dcf']['intrinsic_value']:.2f}
            Market Price: ${fin['dcf']['current_price']:.2f} | Upside/Downside: {fin['dcf']['upside_pct']:+.1f}%
            Revenue Notes: {fin['revenue_explanation']}
            Margin Notes: {fin['margin_explanation']}
            Debt/Leverage Notes: {fin['debt_explanation']}

            [NEWS & SENTIMENT]
            News Summary: {news['summary']}
            Fear & Greed Index: {sent['fear_greed_index']}/100
            Bullish Buzz: {sent['bullish_pct']}% | Bearish Buzz: {sent['bearish_pct']}%

            [QUANTITATIVE RISK]
            Annual Volatility: {risk['metrics']['volatility']*100:.1f}%
            Sharpe Ratio: {risk['metrics']['sharpe']:.2f} | Sortino: {risk['metrics']['sortino']:.2f}
            Beta: {risk['metrics']['beta']:.2f} | Max Drawdown: {risk['metrics']['max_drawdown']*100:.1f}%
            VaR (95%, 1-day): {risk['metrics']['var_95']*100:.2f}%
            Risk Summary: {risk['explanation']}

            [SEC & EARNINGS]
            SEC Filing Summary: {sec['summary']}
            Earnings Outlook: {earn['outlook']}
            CEO Confidence: {earn['ceo_confidence']}%
            """
        except Exception as e:
            print(f"RAG context build error: {e}")

    # Build prompt for LLM
    prompt = f"""
    {rag_context}

    User Question: {msg}

    Respond as Finance Assistant, an institutional-grade financial analyst.
    Answer the user's SPECIFIC question directly using the data provided above.
    - Start with a direct 1-2 sentence answer to their question.
    - Follow with bullet points citing specific numbers from the context.
    - End with: *Remember, AI recommendations do not constitute guaranteed financial advice.*
    Do NOT give a generic overview — answer exactly what was asked.
    """

    llm_output = generate_text(prompt, system_instruction="You are Finance Assistant, an institutional-grade financial analyst. Always answer the user's specific question using the provided data. Never give generic responses.")

    if not llm_output:
        # Smart question-aware fallback using real fetched data
        llm_output = _build_smart_fallback(msg, ticker, stock_data)

    # Save assistant response
    db.add(ChatHistory(session_id=session, role="assistant", content=llm_output))
    db.commit()

    return {"response": llm_output}


def _build_smart_fallback(msg: str, ticker: str, stock_data: dict) -> str:
    """
    Question-aware fallback that routes to the right data based on user intent.
    Used when no LLM API key is available.
    """
    msg_lower = msg.lower()
    t = ticker.upper() if ticker else "this stock"

    if not stock_data:
        return (
            f"I couldn't retrieve live data for **{t}** right now. "
            "Please ensure the backend is running and try again.\n\n"
            "*Remember, AI recommendations do not constitute guaranteed financial advice.*"
        )

    tech = stock_data["agents"]["technical"]
    ind = tech["indicators"]
    fin = stock_data["agents"]["financials"]
    news = stock_data["agents"]["news"]
    sent = stock_data["agents"]["sentiment"]
    risk = stock_data["agents"]["risk"]
    sec = stock_data["agents"]["sec"]
    earn = stock_data["agents"]["earnings"]
    summ = stock_data["summary"]
    price = stock_data["current_price"]
    chg = stock_data["price_change_pct"]

    # --- Intent routing ---

    # Support / Resistance
    if any(w in msg_lower for w in ["support", "resistance", "level", "floor", "ceiling"]):
        return (
            f"**{t}** has a local support level at **${ind['support']:.2f}** and resistance at **${ind['resistance']:.2f}**.\n\n"
            f"- Current price: **${price:.2f}** ({chg:+.2f}% today)\n"
            f"- SMA 20 (medium-term support): **${ind['sma_20']:.2f}**\n"
            f"- SMA 50 (long-term support): **${ind['sma_50']:.2f}**\n"
            f"- Trend channel: **{tech['trend']}**\n\n"
            f"*{tech['summary']}*\n\n"
            "*Remember, AI recommendations do not constitute guaranteed financial advice.*"
        )

    # RSI
    if "rsi" in msg_lower or "overbought" in msg_lower or "oversold" in msg_lower:
        rsi = ind['rsi']
        signal = "overbought (potential pullback zone)" if rsi > 70 else "oversold (potential rebound zone)" if rsi < 30 else "neutral (no extreme signal)"
        return (
            f"**{t}** RSI(14) is currently **{rsi:.1f}** — considered **{signal}**.\n\n"
            f"- MACD Histogram: **{ind['macd_hist']:.4f}** ({'bullish momentum' if ind['macd_hist'] > 0 else 'bearish momentum'})\n"
            f"- Overall trend: **{tech['trend']}**\n"
            f"- Technical score: **{summ['technical_score']:.1f}/10**\n\n"
            f"*{tech['summary']}*\n\n"
            "*Remember, AI recommendations do not constitute guaranteed financial advice.*"
        )

    # MACD
    if "macd" in msg_lower or "momentum" in msg_lower or "crossover" in msg_lower:
        macd = ind['macd_hist']
        return (
            f"**{t}** MACD histogram is **{macd:.4f}** — indicating **{'bullish' if macd > 0 else 'bearish'} momentum**.\n\n"
            f"- RSI(14): **{ind['rsi']:.1f}**\n"
            f"- Trend: **{tech['trend']}**\n"
            f"- Technical score: **{summ['technical_score']:.1f}/10**\n\n"
            f"*{tech['summary']}*\n\n"
            "*Remember, AI recommendations do not constitute guaranteed financial advice.*"
        )

    # DCF / Intrinsic Value / Valuation
    if any(w in msg_lower for w in ["dcf", "intrinsic", "value", "valuation", "worth", "undervalued", "overvalued", "fair"]):
        upside = fin['dcf']['upside_pct']
        direction = "undervalued" if upside > 0 else "overvalued"
        return (
            f"Based on our DCF model, **{t}** is currently **{direction}** by **{abs(upside):.1f}%**.\n\n"
            f"- Market Price: **${fin['dcf']['current_price']:.2f}**\n"
            f"- DCF Intrinsic Value: **${fin['dcf']['intrinsic_value']:.2f}**\n"
            f"- Upside/Downside: **{upside:+.1f}%**\n"
            f"- Financial Score: **{summ['financial_score']:.1f}/10**\n\n"
            f"{fin['dcf_explanation']}\n\n"
            "*Remember, AI recommendations do not constitute guaranteed financial advice.*"
        )

    # Buy / Sell / Recommendation
    if any(w in msg_lower for w in ["buy", "sell", "hold", "recommend", "invest", "should i", "position"]):
        return (
            f"Finance Assistant's multi-agent consensus for **{t}** is: **{summ['recommendation']}** "
            f"(Confidence: **{summ['confidence_pct']}%**, Risk: **{summ['risk_level']}**).\n\n"
            f"- Financial Score: **{summ['financial_score']:.1f}/10**\n"
            f"- Technical Score: **{summ['technical_score']:.1f}/10**\n"
            f"- DCF Upside: **{fin['dcf']['upside_pct']:+.1f}%**\n"
            f"- News Sentiment: **{summ['news_sentiment']}**\n"
            f"- Fear & Greed: **{sent['fear_greed_index']}/100**\n\n"
            "*Remember, AI recommendations do not constitute guaranteed financial advice.*"
        )

    # News / Sentiment
    if any(w in msg_lower for w in ["news", "sentiment", "buzz", "headline", "media", "social"]):
        return (
            f"**{t}** news sentiment is currently **{summ['news_sentiment']}**.\n\n"
            f"- Fear & Greed Index: **{sent['fear_greed_index']}/100**\n"
            f"- Bullish social buzz: **{sent['bullish_pct']}%** | Bearish: **{sent['bearish_pct']}%**\n\n"
            f"**News Summary:** {news['summary']}\n\n"
            "*Remember, AI recommendations do not constitute guaranteed financial advice.*"
        )

    # Risk / Volatility / VaR / Drawdown
    if any(w in msg_lower for w in ["risk", "volatility", "var", "drawdown", "sharpe", "beta", "sortino"]):
        return (
            f"**{t}** quantitative risk profile:\n\n"
            f"- Annual Volatility: **{risk['metrics']['volatility']*100:.1f}%**\n"
            f"- Beta: **{risk['metrics']['beta']:.2f}** (vs. market)\n"
            f"- Sharpe Ratio: **{risk['metrics']['sharpe']:.2f}**\n"
            f"- Sortino Ratio: **{risk['metrics']['sortino']:.2f}**\n"
            f"- Max Drawdown: **{risk['metrics']['max_drawdown']*100:.1f}%**\n"
            f"- Value at Risk (95%, 1-day): **{risk['metrics']['var_95']*100:.2f}%**\n\n"
            f"*{risk['explanation']}*\n\n"
            "*Remember, AI recommendations do not constitute guaranteed financial advice.*"
        )

    # SEC / Legal / Filings
    if any(w in msg_lower for w in ["sec", "legal", "lawsuit", "filing", "regulatory", "risk factor", "10-k"]):
        return (
            f"**{t}** SEC filing analysis:\n\n"
            f"**Filing Summary:** {sec['summary']}\n\n"
            f"**Flagged Risks:**\n" +
            "\n".join(f"- {r}" for r in sec.get('hidden_risks', [])) +
            f"\n\n*Remember, AI recommendations do not constitute guaranteed financial advice.*"
        )

    # Earnings / CEO / Outlook
    if any(w in msg_lower for w in ["earnings", "ceo", "outlook", "guidance", "forecast", "revenue", "profit"]):
        return (
            f"**{t}** earnings intelligence:\n\n"
            f"- CEO Confidence Index: **{earn['ceo_confidence']}%**\n"
            f"- Outlook: {earn['outlook']}\n\n"
            f"**Strategic Plans:**\n" +
            "\n".join(f"- {p}" for p in earn.get('plans', [])) +
            f"\n\n*Remember, AI recommendations do not constitute guaranteed financial advice.*"
        )

    # Price / Performance
    if any(w in msg_lower for w in ["price", "performance", "return", "gain", "loss", "today", "change"]):
        return (
            f"**{t}** is currently trading at **${price:.2f}** ({chg:+.2f}% today).\n\n"
            f"- Recommendation: **{summ['recommendation']}** ({summ['confidence_pct']}% confidence)\n"
            f"- DCF Intrinsic Value: **${fin['dcf']['intrinsic_value']:.2f}** ({fin['dcf']['upside_pct']:+.1f}% upside)\n"
            f"- 52W Support: **${ind['support']:.2f}** | Resistance: **${ind['resistance']:.2f}**\n\n"
            "*Remember, AI recommendations do not constitute guaranteed financial advice.*"
        )

    # Generic fallback — still uses real data unlike before
    return (
        f"Here is the Finance Assistant summary for **{t}**:\n\n"
        f"- Price: **${price:.2f}** ({chg:+.2f}% today)\n"
        f"- Consensus: **{summ['recommendation']}** | Confidence: **{summ['confidence_pct']}%**\n"
        f"- Technical trend: **{tech['trend']}** | RSI: **{ind['rsi']:.1f}**\n"
        f"- DCF upside: **{fin['dcf']['upside_pct']:+.1f}%** | Financial score: **{summ['financial_score']:.1f}/10**\n"
        f"- News sentiment: **{summ['news_sentiment']}** | Fear & Greed: **{sent['fear_greed_index']}/100**\n\n"
        f"Feel free to ask more specifically about support levels, RSI, DCF valuation, risk metrics, SEC filings, or earnings.\n\n"
        "*Remember, AI recommendations do not constitute guaranteed financial advice.*"
    )
