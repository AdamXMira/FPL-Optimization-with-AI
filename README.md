COMP 401 Final Project Repository

This repository contains a complete machine learning and linear programming pipeline to optimize Fantasy Premier League (FPL) team selection.

⚖️ Project Summary

Fantasy Premier League (FPL) is a resource allocation problem under strict constraints such as budget, squad size, and positional constraints. This project treats FPL team selection as a binary linear program, using the following:

- XGBoost: to predict each player's FPL performance for the 2025-2026 season based on historical data.

- PuLP: to select the optimal squad of players that maximizes predicted points under FPL constraints.

⚛️ Optimization Problem

A linear program seeks to maximize or minimize a linear objective function subject to linear equality and inequality constraints. This project utilizes:

- PuLP: an open-source Python LP library with CBC and Gurobi solvers which are the most efficient/quick.

- Binary Decision Variables: select or ignore each player (either 1 or 0).

The objective is to maximize total predicted points for the entire season while satisfying:

- Budget limit 

- Positional constraints (e.g., 1 GK, 3-5 DEF, etc.)

- Average minutes filter

🏆 Model Overview

An XGBoost Regressor is trained using player data from the previous two FPL seasons to predict total points for the upcoming 2025–2026 season.

Assumptions:

- No transfers

- No substitutions

- Players remain fit all season

- Predicted points are exact


Advantages:

- Unlike heuristic-based models, our predictor does not assume repeat scores. Instead, it learns from the historical performances across all relevant statistics and makes accurate predictions based on them.

📊 Model Training Notes

Training Data: Cleaned player stats from 2023-2024 and 2024-2025 seasons

Excluded Features:

- full_name: categorical label only

- season: used only for season weighting

- now_cost: used only in optimization, not prediction

- ownership: not predictive; reflects popularity not performance

✨ Interactive Streamlit App

An interactive Streamlit-based GUI allows:

- Budget customization

- Positional flexibility

- Average minutes constraint

- Future GUI enhancements will include player info dropdowns and visual team formations.

⚒️ Future Work

⚡ SHAP analysis for model explainability

⚡ Scraping player clubs to implement same-club restrictions

⚡ Deep learning regressors (e.g., using xG, npxG)

⚡ Optuna for hyperparameter tuning

⚡ Bayesian optimization for dynamic season weight tuning

⚡ Real-time updates via live PL API

⚡ Classification models for match result predictions

⚡ Chip strategy integration (bench boost, triple captain, wildcard)


📅 Assumptions and Simplifications

- No real-time data updates

- No new players from other leagues modeled (yet)

- Players expected to play full season

- One-time selection without gameweek-specific strategies


✅ Key Features

- Data-driven team selection

- Transparent model design and assumptions

- User freedom via GUI for personalized constraints

- Markdown reports and downloadable CSV outputs

- Clean, customizable design and styling


🔍 Usage Instructions

1- Clone the repo

2- Install dependencies (pip install -r requirements.txt)

3- Run train_model.py to train and save predictions

4- Launch Streamlit GUI: streamlit run Gui.py


🔧 Tech used

- Python 3.9+

- pandas, xgboost, scikit-learn

- PuLP (with CBC solver)

- Streamlit for GUI

🚀 Credits

Developed by Adam Mira for COMP 401 (Kenyon College) with the help of Prof. James Skon and Prof. Erin Leatherman.