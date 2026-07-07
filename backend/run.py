import uvicorn
import os

if __name__ == "__main__":
    # Ensure port is an integer
    port = int(os.getenv("PORT", 8000))
    print(f"Starting Finance Assistant Backend on http://localhost:{port}...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
