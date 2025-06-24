
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Football Player Recommendation", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    df = pd.read_csv("dataset_BI5.csv")
    clusters = pd.read_csv("KMeans_Cluster_Profiles.csv")
    return df, clusters

df, cluster_profiles = load_data()

# Sidebar filters
st.sidebar.title("🔎 Filter Players")
age_range = st.sidebar.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (20, 35))
leagues = st.sidebar.multiselect("League", df["Competition"].unique(), default=list(df["Competition"].unique()))
positions = st.sidebar.multiselect("Position", df["Position"].unique(), default=list(df["Position"].unique()))
cluster_options = sorted(df["KMeans_Cluster"].dropna().unique())
clusters = st.sidebar.multiselect("Cluster", cluster_options, default=cluster_options)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Similar Players", "📊 Radar Chart", "🧭 Cluster Explorer", "ℹ️ About"])

with tab1:
    st.header("🔍 Similar Player Search")
    filtered = df[
        (df["Age"].between(*age_range)) &
        (df["Competition"].isin(leagues)) &
        (df["Position"].isin(positions)) &
        (df["KMeans_Cluster"].isin(clusters))
    ]
    player_list = sorted(filtered["Player"].unique())
    selected_player = st.selectbox("Choose a player", player_list)

    # Similarity
    feature_cols = [col for col in df.columns if df[col].dtype in [float, int] and col not in ["Age", "KMeans_Cluster"]]
    selected_vector = filtered[filtered["Player"] == selected_player][feature_cols].values
    if selected_vector.shape[0] > 0:
        similarity_scores = cosine_similarity(selected_vector, filtered[feature_cols]).flatten()
        filtered["Similarity"] = similarity_scores
        result = filtered.sort_values("Similarity", ascending=False).head(10)
        st.dataframe(result[["Player", "Similarity", "Age", "Competition", "Position", "KMeans_Cluster"]], use_container_width=True)
        st.download_button("Download Results", result.to_csv(index=False), file_name="similar_players.csv")

with tab2:
    st.header("📊 Radar Chart Comparison")
    p1 = st.selectbox("Player 1", df["Player"].unique(), index=0)
    p2 = st.selectbox("Player 2", df["Player"].unique(), index=1)

    p1_vals = df[df["Player"] == p1][feature_cols].mean()
    p2_vals = df[df["Player"] == p2][feature_cols].mean()

    radar_df = pd.DataFrame({
        "Metric": feature_cols,
        p1: p1_vals.values,
        p2: p2_vals.values
    }).melt(id_vars="Metric", var_name="Player", value_name="Z-Score")

    fig = px.line_polar(radar_df, r="Z-Score", theta="Metric", color="Player", line_close=True)
    fig.update_traces(fill="toself")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🧭 Explore Cluster Profiles")
    cluster_id = st.selectbox("Select Cluster", sorted(cluster_profiles["KMeans_Cluster"].unique()))
    row = cluster_profiles[cluster_profiles["KMeans_Cluster"] == cluster_id].drop(columns="KMeans_Cluster").T.reset_index()
    row.columns = ["Metric", "Z-Score"]
    fig = px.line_polar(row, r="Z-Score", theta="Metric", line_close=True, title=f"Cluster {cluster_id}")
    fig.update_traces(fill="toself")
    st.plotly_chart(fig, use_container_width=True)

    players = df[df["KMeans_Cluster"] == cluster_id][["Player", "Age", "Competition", "Position"]]
    st.write(f"Players in Cluster {cluster_id}:")
    st.dataframe(players, use_container_width=True)
    st.download_button("Download Player List", players.to_csv(index=False), file_name=f"cluster_{cluster_id}_players.csv")

with tab4:
    st.header("ℹ️ About This Dashboard")
    st.markdown("""
This tool was developed as part of a Master's thesis project to support football recruitment.

**Features:**
- Role-based clustering with KMeans (k=7)
- Per 90-minute standardized stats
- Big 5 Leagues (2023–24)
- Similarity search using cosine similarity
- Radar comparisons and cluster explorer

**Made with**: Streamlit, pandas, scikit-learn, plotly
""")
