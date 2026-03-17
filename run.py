"""
Entry point — run the Flask application.
Usage: python run.py
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))
from routes import app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
