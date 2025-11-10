# 🏎️  /🏁 F1 Season Analytics (Streamlit App)

This project is an **interactive Formula 1 analytics dashboard** built with **Streamlit**, **pandas**, and **matplotlib/seaborn**.  
It uses the [Ergast/Jolpi F1 API](http://ergast.com/mrd/) to  race, qualifying, and sprint results, then provides **visual insights** into drivers’ and constructors’ performance.

---

##  Features

- 📊 **Standings (Drivers & Constructors)**
  - Official vs. computed points (race + sprint)
  - Extra metrics: podiums, pole positions, wins
  - Clean tables without technical IDs
    <img src="image/standings.png" width="700"/>
- 📈 **Cumulative Points**
  - Interactive line chart of drivers’ cumulative points
  - User selects which drivers to display
    <img src="image/cumulative.png" width="700"/>
- 🎯 **Result Qualifying vs. Race**
  - Scatter plot of average quali vs. race positions
  - Team-based colors, diagonal reference line
  - Shows who gains/loses places on Sundays
    <img src="image/quali_vs_race_1.png" width="700"/>    <img src="image/quali_vs_race_2.png" width="700"/>
- 🏗️ **Constructors**
  - Stacked bars of points contribution per driver
  - Pie chart of constructors’ points share
    <img src="image/constructor.png" width="700"/>

- 🏆 **Consistency**
  - Side-by-side boxplots of qualifying & race results
  - Shows performance variability per driver
    <img src="image/consistence.png" width="700"/>

- ⏱️ **Session Analysis**
  - Standing of the session, with rank and best lap time
    <img src="image/data_about_session_1.png" width="700"/>
  - Labs analysis per driver
    <img src="image/recent_lab.png" width="700"/>
  - Stint analysis with pit stop annotations
    <img src="image/stint_analysis.png" width="700"/>
  - Lap time delta analysis vs. a reference driver
    <img src="image/delta_time.png" width="700"/>  
- 🏎️ **Current driver form**: Selection of a period and drivers
 <img src="image/current_from_1.png" width="700"/>
  <img src="image/current_from_2.png" width="700"/>
---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/f1-season-analytics.git
cd f1-season-analytics
