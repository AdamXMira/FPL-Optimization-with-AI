import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, LpStatus

# Load prediction data 
df = pd.read_csv("player_predictions.csv")

#  Basic sanity checks 
df = df.dropna(subset=['full_name', 'predicted_points', 'position', 'now_cost', 'minutes'])
df['now_cost'] = df['now_cost'].astype(float)

position_map = {0: 'GK', 1: 'DEF', 2: 'MID', 3: 'FWD'}
if pd.api.types.is_numeric_dtype(df['position']):
    df['position'] = df['position'].map(position_map)

# UI
# Set up the page title and layout 
st.set_page_config(page_title="FPL Optimizer", layout="centered")
st.markdown("<h1 style='text-align: center;'>⚽ Interactive FPL Squad Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Choose your preferences and let our model pick the best FPL team for you!.</p>", unsafe_allow_html=True)

# Set up the background image and style
# This CSS code sets the background image, styles the container, and adds hover effects to buttons and sliders
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://cdn.vox-cdn.com/thumbor/aHCMHJOho2xbVwq8cmGM16pPlng=/0x0:3706x2383/1820x1213/filters:focal(2125x500:2717x1092):format(webp)/cdn.vox-cdn.com/uploads/chorus_image/image/72017797/1247532740.0.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    .block-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 0.5rem;
    }

    h1, p {
        color: #000000;
    }

    .stButton > button:hover {
        background-color: #f63366 !important;
        color: white !important;
        transition: 0.3s ease-in-out;
    }

    .stSlider:hover, .stMarkdown:hover {
        background-color: rgba(255,255,255,0.05);
        transition: 0.3s ease-in-out;
    }

    .stExpander:hover {
        background-color: rgba(255, 245, 230, 0.5);
        transition: 0.3s ease-in-out;
        border-left: 3px solid #f63366;
    }
    </style>

    """,
    unsafe_allow_html=True
)


# Header
st.header("🔧 Constraints")
# This section allows the user to set constraints for the optimization
player_count = st.slider("Number of Players", 7, 15, 11, step=1, help="Total number of players in squad")
budget = st.slider("Your Maximum Budget (£M)", 40.0, 100.0, 83.0, step=0.5, help="Total budget for the squad")
min_avg_minutes = st.slider("Minimum Average Minutes Played", 0, 3000, 1000, step=10, help="Minimum average minutes played across selected players")



# Optimize Button
# This button triggers the optimization process
# When clicked, it will run the optimization model and display the results
if st.button("🔍 Optimize Squad"):
    model = LpProblem("FPL_Optimizer", LpMaximize)
    player_vars = LpVariable.dicts("Select", df.index, cat=LpBinary)

    # Objective: maximize predicted points
    model += lpSum(player_vars[i] * df.loc[i, 'predicted_points'] for i in df.index)

    # Budget constraint (cost is stored as tenths)
    model += lpSum(player_vars[i] * df.loc[i, 'now_cost'] for i in df.index) <= budget * 10

    # Squad size constraint
    model += lpSum(player_vars[i] for i in df.index) == player_count

    # Position constraints
    # Ensure at least 1 GK, 3 DEF, 2 MID, and 1 FWD
    # and at most 2 GK, 5 DEF, 5 MID, and 3 FWD
    # The position constraints are based on the FPL rules
    model += lpSum(player_vars[i] for i in df.index if df.loc[i, 'position'] == 'GK') >=1
    model += lpSum(player_vars[i] for i in df.index if df.loc[i, 'position'] == 'GK') <= 2
    model += lpSum(player_vars[i] for i in df.index if df.loc[i, 'position'] == 'DEF') >= 3
    model += lpSum(player_vars[i] for i in df.index if df.loc[i, 'position'] == 'DEF') <= 5
    model += lpSum(player_vars[i] for i in df.index if df.loc[i, 'position'] == 'MID') >= 2
    model += lpSum(player_vars[i] for i in df.index if df.loc[i, 'position'] == 'MID') <= 5
    model += lpSum(player_vars[i] for i in df.index if df.loc[i, 'position'] == 'FWD') >= 1
    model += lpSum(player_vars[i] for i in df.index if df.loc[i, 'position'] == 'FWD') <= 3

    # Average minutes constraint (converted to total minutes for 11 players)
    model += lpSum(player_vars[i] * df.loc[i, 'minutes'] for i in df.index) >= min_avg_minutes * player_count

    # Solve
    model.solve()
    # Check the status of the optimization
    # If the optimization is not successful, display an error message
    # If the optimization is successful, display the optimized squad
    if LpStatus[model.status] != 'Optimal':
        st.error(f"Optimization failed, please edit your constraints: {LpStatus[model.status]}")
    else:
        selected = df[[player_vars[i].varValue == 1 for i in df.index]].copy()
        selected = selected.sort_values(by='predicted_points', ascending=False)

        st.subheader("Optimized Squad")
        st.dataframe(
            selected[['full_name', 'position', 'now_cost', 'predicted_points', 'minutes']]
            .assign(
                now_cost=lambda df: df['now_cost'] / 10  # Convert to £M
            )
            .rename(columns={
                'full_name': 'Player',
                'position': 'Position',
                'now_cost': 'Predicted Cost (£M)',
                'predicted_points': 'Predicted Pts',
                'minutes': 'Minutes This Season'
            }),
            hide_index=True
        )
        total_cost = selected['now_cost'].sum() / 10
        total_points = selected['predicted_points'].sum()
        avg_minutes = selected['minutes'].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("💸 Total Cost", f"£{total_cost:.1f}M")
        col2.metric("📈 Total Points (Without Captaincy)", f"{total_points:.2f}")
        col3.metric("⏱️ Avg Minutes", f"{avg_minutes:.0f}")

        selected.to_csv("interactive_optimized_squad.csv", index=False)
        st.download_button(
        label="📥 Download Squad CSV",
        data=selected.to_csv(index=False),
        file_name="optimized_squad.csv",
        mime="text/csv"
)
