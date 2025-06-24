# Streamlit Player Recommendation Dashboard

This is a prototype of a football scouting dashboard for stylistic player recommendations based on role clustering and performance metrics.

## 🚀 How to Run

1. **Install dependencies** (Python 3.9+ recommended):
```
pip install -r requirements.txt
```

2. **Start the app**:
```
streamlit run app.py
```

3. **Interact**:
- Use the tabs to explore:
  - Similarity search (e.g., find Kroos replacements)
  - Radar chart comparisons
  - Cluster profiles and player lists

## 📂 Files Included
- `dataset_BI5.csv`: Full Big 5 player dataset (standardized, per 90, with metadata and clusters)
- `KMeans_Cluster_Profiles.csv`: Average stats per cluster (used for radar charts)
- `toni_kroos_similarity_results.csv`: Example output
- `app.py`: Streamlit dashboard code
- `requirements.txt`: Python dependencies

---

## 🧠 Methodology

- Data: FBref + StatsBomb, Big 5 Leagues (2023–24), >900 minutes, per-90 stats
- Clustering: KMeans (k=7) to define player roles
- Similarity: Cosine similarity on standardized performance vectors
- Tools: Streamlit, pandas, scikit-learn, matplotlib, plotly