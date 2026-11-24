import sys
import subprocess

def print_menu():
    print("=" * 60)
    print("       CreditWise AI - Loan Approval Application")
    print("=" * 60)
    print("1. Start FastAPI Backend + Web UI (http://127.0.0.1:8000)")
    print("2. Start Streamlit Interactive Dashboard (http://localhost:8501)")
    print("3. Exit")
    print("=" * 60)

if __name__ == "__main__":
    print_menu()
    choice = input("Enter choice (1, 2, or 3): ").strip()
    if choice == "1":
        print("\nStarting FastAPI & UI on http://127.0.0.1:8000 ...")
        subprocess.run([sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000", "--reload"])
    elif choice == "2":
        print("\nStarting Streamlit on http://localhost:8501 ...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    else:
        print("Exiting.")
