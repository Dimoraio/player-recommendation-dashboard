import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Football Player Recommendation", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("dataset_BI5.csv")
    clusters = pd.read_csv("KMeans_Cluster_Profiles.csv")
    return df, clusters

df, cluster_profiles = load_data()

# Sidebar Filters
st.sidebar.header("Filter Players")
age_range = st.sidebar.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (20, 35))
leagues = st.sidebar.multiselect("Select League(s)", options=df["Competition"].unique(), default=list(df["Competition"].unique()))
positions = st.sidebar.multiselect("Select Position(s)", options=df["Position"].unique(), default=list(df["Position"].unique()))
clusters = st.sidebar.multiselect("Select Cluster(s)", options=sorted(df["KMeans_Cluster"].unique()), default=sorted(df["KMeans_Cluster"].unique()))

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Similarity Search", "📊 Radar Comparison", "🧭 Cluster Explorer", "ℹ️ About"])

with tab1:
    st.title("🔍 Player Similarity Search")
    filtered = df[
        (df["Age"].between(*age_range)) &
        (df["Competition"].isin(leagues)) &
        (df["Position"].isin(positions)) &
        (df["KMeans_Cluster"].isin(clusters))
    ]
    player_list = sorted(filtered["Player"].unique())
    selected_player = st.selectbox("Select a Player", player_list)

    features = [col for col in df.columns if "Per_90" in col or "Expected" in col or "Progression" in col or "Performance" in col]
    selected_vector = filtered[filtered["Player"] == selected_player][features].values
    if selected_vector.shape[0] > 0:
        sims = cosine_similarity(selected_vector, filtered[features]).flatten()
        filtered["Similarity"] = sims
        results = filtered.sort_values(by="Similarity", ascending=False).head(10)[["Player", "Similarity", "Age", "Competition", "Position", "KMeans_Cluster"]]
        st.write(f"Top stylistic matches for **{selected_player}**:")
        st.dataframe(results, use_container_width=True)
        st.download_button("Download CSV", results.to_csv(index=False), file_name="similar_players.csv")

with tab2:
    st.title("📊 Radar Chart Comparison")
    player1 = st.selectbox("Player 1", df["Player"].unique(), index=0, key="p1")
    player2 = st.selectbox("Player 2", df["Player"].unique(), index=1, key="p2")

    def radar_data(name):
        return df[df["Player"] == name][features].mean()

    if player1 and player2:
        d1 = radar_data(player1)
        d2 = radar_data(player2)
        radar_df = pd.DataFrame({"Metric": features, player1: d1.values, player2: d2.values})
        radar_df = pd.melt(radar_df, id_vars="Metric", var_name="Player", value_name="Z-Score")
        fig = px.line_polar(radar_df, r="Z-Score", theta="Metric", color="Player", line_close=True)
        fig.update_traces(fill="toself")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.title("🧭 Cluster Explorer")
    selected_cluster = st.selectbox("Select a Cluster", sorted(cluster_profiles["KMeans_Cluster"].unique()))
    row = cluster_profiles[cluster_profiles["KMeans_Cluster"] == selected_cluster].drop(columns="KMeans_Cluster").T.reset_index()
    row.columns = ["Metric", "Z-Score"]
    fig = px.line_polar(row, r="Z-Score", theta="Metric", line_close=True, title=f"Cluster {selected_cluster} Profile")
    fig.update_traces(fill="toself")
    st.plotly_chart(fig, use_container_width=True)

    players = df[df["KMeans_Cluster"] == selected_cluster][["Player", "Age", "Competition", "Position"]]
    st.write(f"Players in Cluster {selected_cluster}:")
    st.dataframe(players, use_container_width=True)
    st.download_button("Download Player List", players.to_csv(index=False), file_name=f"cluster_{selected_cluster}_players.csv")

with tab4:
    st.title("ℹ️ About This Project")
    st.markdown("""
    This dashboard was created as part of a Master's project to support football recruitment using data.

    **Key Features**:
    - Similarity search using cosine similarity
    - Clustering using KMeans (k=7)
    - Per 90-minute standardized metrics
    - Big 5 European leagues (2023–24)

    **Developed with**: Streamlit, scikit-learn, pandas, plotly
    """)