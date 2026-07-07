from app.database import init_db, SessionLocal, SavedPortfolio

def test_db_saved_portfolio():
    init_db()
    db = SessionLocal()
    try:
        # Create a saved portfolio
        portfolio = SavedPortfolio(
            capital=10000.0,
            risk_appetite="Moderate",
            horizon=5,
            allocation={"SPY": {"weight": 40, "ticker": "SPY"}}
        )
        db.add(portfolio)
        db.commit()
        
        portfolio_id = portfolio.id
        
        # Close session and open a fresh one to test database persistence
        db.close()
        db = SessionLocal()
        
        saved = db.query(SavedPortfolio).filter(SavedPortfolio.id == portfolio_id).first()
        assert saved is not None
        # Assert that capital is correct. Note: if the bug is present, capital won't be saved in DB
        assert saved.capital == 10000.0
        assert saved.risk_appetite == "Moderate"
        assert saved.horizon == 5
    finally:
        db.close()
