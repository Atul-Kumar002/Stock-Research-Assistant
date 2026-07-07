from app.agents.orchestrator import generate_debate_transcript


def test_generate_debate_transcript_handles_missing_fields():
    agents = {
        "technical": {"trend": "Bullish", "score": 7.5, "indicators": {"rsi": 62.0, "support": 100.0}},
        "financials": {"score": 7.0, "dcf": {"upside_pct": 12.5}},
        "risk": {"risk_level": "Medium", "metrics": {"volatility": 0.2, "max_drawdown": 0.1}},
        "macro": {},
    }
    rec = {"recommendation": "BUY", "confidence_pct": 80}

    transcript = generate_debate_transcript("AAPL", agents, rec)

    assert len(transcript) == 5
    assert transcript[0]["agent"] == "Technical Analysis Agent"
    assert transcript[-1]["verdict"] == "BUY"
