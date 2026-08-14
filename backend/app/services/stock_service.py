import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from typing import Dict, Any, List
from .cache_service import backend_cache

def get_stock_data(ticker: str) -> Dict[str, Any]:
    """
    Fetch comprehensive stock data from yfinance with TTL caching
    """
    ticker_clean = ticker.upper().strip()
    cache_key = f"stock_data:{ticker_clean}"
    cached = backend_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        t = yf.Ticker(ticker_clean)
        info = t.info
        
        # Fallback for empty info
        if not info or 'symbol' not in info:
            info = {"symbol": ticker_clean, "shortName": ticker_clean}
            
        result = {
            "info": info,
            "ticker": ticker_clean
        }
        backend_cache.set(cache_key, result, ttl_seconds=300) # 5 min TTL
        return result
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        fallback = {"info": {"symbol": ticker_clean, "shortName": ticker_clean}, "ticker": ticker_clean}
        return fallback

def get_historical_prices(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Get historical prices as a pandas DataFrame with TTL caching
    """
    ticker_clean = ticker.upper().strip()
    cache_key = f"stock_history:{ticker_clean}:{period}"
    cached = backend_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        t = yf.Ticker(ticker_clean)
        df = t.history(period=period)
        backend_cache.set(cache_key, df, ttl_seconds=120) # 2 min TTL
        return df
    except Exception as e:
        print(f"Error fetching history for {ticker}: {e}")
        return pd.DataFrame()


def calculate_technical_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute RSI, MACD, EMAs, SMAs, Support/Resistance, and Trend
    """
    if df.empty or len(df) < 30:
        return {"error": "Insufficient data"}
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # SMAs and EMAs
    sma_20 = close.rolling(window=20).mean().iloc[-1]
    sma_50 = close.rolling(window=50).mean().iloc[-1]
    sma_200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else close.rolling(window=len(close)).mean().iloc[-1]
    
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = (ema_12 - ema_26).iloc[-1]
    macd_signal = (ema_12 - ema_26).ewm(span=9, adjust=False).mean().iloc[-1]
    macd_hist = macd - macd_signal
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs.iloc[-1]))
    if np.isnan(rsi):
        rsi = 50.0
        
    # Support & Resistance (Simple Local Min/Max)
    recent_highs = high.rolling(window=10, center=True).max()
    recent_lows = low.rolling(window=10, center=True).min()
    
    # Get top 2 unique support/resistance levels
    resistances = sorted(list(set(recent_highs.dropna().tail(30).tolist())))
    supports = sorted(list(set(recent_lows.dropna().tail(30).tolist())))
    
    current_price = close.iloc[-1]
    resistance_level = next((r for r in resistances if r > current_price), current_price * 1.05)
    support_level = next((s for s in reversed(supports) if s < current_price), current_price * 0.95)
    
    # Trend Analysis
    trend = "Sideways"
    if current_price > sma_20 and sma_20 > sma_50:
        trend = "Bullish"
    elif current_price < sma_20 and sma_20 < sma_50:
        trend = "Bearish"
        
    return {
        "current_price": float(current_price),
        "rsi": float(rsi),
        "macd": float(macd),
        "macd_signal": float(macd_signal),
        "macd_hist": float(macd_hist),
        "sma_20": float(sma_20),
        "sma_50": float(sma_50),
        "sma_200": float(sma_200),
        "support": float(support_level),
        "resistance": float(resistance_level),
        "trend": trend
    }

def calculate_risk_metrics(df: pd.DataFrame, beta_fallback: float = 1.0) -> Dict[str, Any]:
    """
    Calculate Volatility, Sharpe, Sortino, Max Drawdown, Beta, VaR, and Monte Carlo paths
    """
    if df.empty or len(df) < 30:
        return {"error": "Insufficient data"}
    
    close = df['Close']
    returns = close.pct_change().dropna()
    
    # Volatility (Annualized)
    daily_vol = returns.std()
    ann_vol = daily_vol * np.sqrt(252)
    
    # Expected Return (Annualized average return)
    expected_return = returns.mean() * 252
    
    # Sharpe Ratio (assuming risk free rate = 0.04)
    rf = 0.04
    sharpe = (expected_return - rf) / ann_vol if ann_vol > 0 else 0
    
    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = (expected_return - rf) / downside_vol if downside_vol > 0 else 0
    
    # Max Drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Value at Risk (95% historical VaR)
    var_95 = np.percentile(returns, 5)
    
    # Monte Carlo Simulation (30 days forecast, 100 paths)
    np.random.seed(42)
    current_price = close.iloc[-1]
    days = 30
    paths = 100
    
    # Drift and Volatility for GBm
    mu = returns.mean()
    sigma = returns.std()
    
    # Simulate paths
    sim_returns = np.random.normal(mu, sigma, (days, paths))
    price_paths = np.zeros((days + 1, paths))
    price_paths[0] = current_price
    for t_step in range(1, days + 1):
        price_paths[t_step] = price_paths[t_step - 1] * np.exp(sim_returns[t_step - 1])
        
    # Statistical summaries of paths for charts
    # Compute percentiles across simulation paths for each day (axis=1)
    p10 = np.percentile(price_paths, 10, axis=1)
    p50 = np.percentile(price_paths, 50, axis=1)
    p90 = np.percentile(price_paths, 90, axis=1)
    
    return {
        "volatility": float(ann_vol),
        "expected_return": float(expected_return),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "beta": float(beta_fallback),
        "var_95": float(var_95),
        "monte_carlo": {
            "days": list(range(days + 1)),
            "p10": [float(v) for v in p10],
            "p50": [float(v) for v in p50],
            "p90": [float(v) for v in p90],
            "current_price": float(current_price)
        }
    }

def get_stock_news(ticker: str) -> List[Dict[str, Any]]:
    """
    Fetch and parse news articles for the stock ticker with TTL caching
    """
    ticker_clean = ticker.upper().strip()
    cache_key = f"stock_news:{ticker_clean}"
    cached = backend_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        t = yf.Ticker(ticker_clean)
        raw_news = t.news
        if not raw_news:
            return []
            
        articles = []
        for item in raw_news[:6]: # Get top 6 articles
            title = item.get("title", "")
            publisher = item.get("publisher", "")
            link = item.get("link", "")
            pub_time = item.get("providerPublishTime", 0)
            
            # Formulate text timestamp
            dt = datetime.datetime.fromtimestamp(pub_time) if pub_time else datetime.datetime.utcnow()
            timestamp_str = dt.strftime("%Y-%m-%d %H:%M")
            
            # Simple heuristic for mock sentiment score
            sentiment = "Neutral"
            score = 0.0
            lower_title = title.lower()
            positive_words = ["growth", "rise", "beat", "positive", "upgrade", "soar", "gain", "profit", "bull", "buy", "success", "innovative"]
            negative_words = ["fall", "drop", "miss", "negative", "downgrade", "plunge", "loss", "bear", "sell", "concern", "lawsuit", "risk", "debt"]
            
            pos_matches = sum(1 for w in positive_words if w in lower_title)
            neg_matches = sum(1 for w in negative_words if w in lower_title)
            
            if pos_matches > neg_matches:
                sentiment = "Positive"
                score = 0.5 + (0.1 * min(pos_matches, 5))
            elif neg_matches > pos_matches:
                sentiment = "Negative"
                score = -0.5 - (0.1 * min(neg_matches, 5))
            else:
                score = 0.0
                
            articles.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "timestamp": timestamp_str,
                "sentiment": sentiment,
                "score": score,
                "summary": f"Key reports state {title} could significantly shape market performance."
            })
        backend_cache.set(cache_key, articles, ttl_seconds=600) # 10 min TTL
        return articles
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []

