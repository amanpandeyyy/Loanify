import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
try:
    os.chdir(BASE_DIR)
except Exception:
    pass

import uvicorn
from app import app

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  🚀 CreditWise AI — Loan Approval Engine is LIVE!")
    print("  👉 Web Dashboard: http://127.0.0.1:8000")
    print("  👉 Swagger API:   http://127.0.0.1:8000/docs")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
