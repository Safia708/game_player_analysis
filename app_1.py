import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from scipy import stats

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, r2_score, mean_absolute_error, silhouette_score
)

try:
    import statsmodels.api  # noqa: F401
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

RANDOM_STATE = 42

REQUIRED_COLS = [
    "player_id", "age", "gender", "game_genre", "difficulty_preference",
    "sessions_per_week", "avg_session_minutes", "playtime_hours_total",
    "player_level", "achievements_unlocked", "matches_played",
    "win_ratio", "kd_ratio", "engagement_level",
    "in_game_purchases", "purchase_amount_usd",
]

st.set_page_config(page_title="Gaming Player Analytics", page_icon="🎮", layout="wide")
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.figsize"] = (9, 5)

st.title("🎮 Gaming Player Analytics")
st.caption(
    "Churn risk, super-fans, spend vs. loyalty, genre/difficulty retention, "
    "player segments, predictive models & skill curves."
)

def get_sample_template() -> pd.DataFrame:
    """Generates a standard sample template with required columns."""
    sample_data = {
        "player_id": ["P1001", "P1002"],
        "age": [24, 30],
        "gender": ["Male", "Female"],
        "game_genre": ["Action", "RPG"],
        "difficulty_preference": ["Medium", "Hard"],
        "sessions_per_week": [5, 12],
        "avg_session_minutes": [45, 90],
        "playtime_hours_total": [120, 450],
        "player_level": [15, 42],
        "achievements_unlocked": [10, 35],
        "matches_played": [80, 310],
        "win_ratio": [0.55, 0.68],
        "kd_ratio": [1.2, 2.1],
        "engagement_level": ["Medium", "High"],
        "in_game_purchases": [1, 1],
        "purchase_amount_usd": [19.99, 89.50],
    }
    return pd.DataFrame(sample_data)


@st.cache_data(show_spinner="Loading file...")
def load_data(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

st.sidebar.header("⚙️ Data Management Options")

action_option = st.sidebar.selectbox(
    "Select Action / Option",
    [
        "📄 Download File Format",
        "📥 Import File",
        "📤 Export File"
    ],
    index=1,
    help="Choose whether you want to download standard format template, import data, or export dataset."
)

if action_option == "📄 Download File Format":
    st.header("📄 Standard File Format Template")
    st.info("Aap yahan se standard gaming player analytics file format template download kar sakte hain.")

    sample_df = get_sample_template()

    st.subheader("Standard Column Schema")
    st.dataframe(sample_df, use_container_width=True)

    # Prepare CSV and Excel downloads
    csv_sample = sample_df.to_csv(index=False).encode("utf-8")
    excel_sample_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_sample_buffer, engine="openpyxl") as writer:
        sample_df.to_excel(writer, index=False, sheet_name="template")
    excel_sample_bytes = excel_sample_buffer.getvalue()

    col_d1, col_d2 = st.columns(2)
    col_d1.download_button(
        "⬇️ Download CSV Format Template",
        data=csv_sample,
        file_name="gaming_player_stats_template.csv",
        mime="text/csv",
        type="primary"
    )
    col_d2.download_button(
        "⬇️ Download Excel Format Template",
        data=excel_sample_bytes,
        file_name="gaming_player_stats_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )
    st.stop()

uploaded_file = None
if action_option in ["📥 Import File", "📤 Export File"]:
    uploaded_file = st.sidebar.file_uploader(
        "📥 Upload File (.xlsx, .xls, .csv)", type=["xlsx", "xls", "csv"]
    )

if action_option == "📥 Import File":
    if uploaded_file is None:
        st.info("👈 Please upload your data file in the sidebar to proceed.")
        st.stop()

    df_uploaded = load_data(uploaded_file)

    # Verify if all required columns exist in the uploaded file
    missing_cols = [col for col in REQUIRED_COLS if col not in df_uploaded.columns]

    if missing_cols:
        st.error("❌ **Invalid File Format!**")
        st.write(
            f"Aap ki uploaded file `{uploaded_file.name}` hamarey required standard format se match **nahi** karti."
        )
        st.warning(f"**Missing Columns ({len(missing_cols)}):** " + ", ".join([f"`{c}`" for c in missing_cols]))

        st.markdown("---")
        st.subheader("🔄 Please download the correct File Format template:")
        
        sample_df = get_sample_template()
        csv_sample = sample_df.to_csv(index=False).encode("utf-8")
        
        st.download_button(
            "📄 Download Standard File Format Template",
            data=csv_sample,
            file_name="gaming_player_stats_template.csv",
            mime="text/csv",
            type="primary"
        )
        st.info("💡 Standard format file download karke usme apna data arrange karen aur dobara upload karen.")
        st.stop()
    else:
        st.sidebar.success("✅ File format verified successfully!")
        df_raw = df_uploaded

if action_option == "📤 Export File":
    if uploaded_file is None:
        st.warning("⚠️ Export karne ke liye pehle file import karen.")
        st.stop()

    df_raw = load_data(uploaded_file)
    st.header("📤 Export Standardized Data")
    st.dataframe(df_raw.head(20), use_container_width=True)

    csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df_raw.to_excel(writer, index=False, sheet_name="data")
    excel_bytes = excel_buffer.getvalue()

    exp1, exp2 = st.columns(2)
    exp1.download_button(
        "⬇️ Download CSV File", data=csv_bytes,
        file_name="gaming_player_stats_export.csv", mime="text/csv", type="primary"
    )
    exp2.download_button(
        "⬇️ Download Excel File", data=excel_bytes,
        file_name="gaming_player_stats_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )
    st.stop()

@st.cache_data(show_spinner="Running analysis pipeline...")
def run_pipeline(df_in: pd.DataFrame):
    df = df_in.copy()
    artifacts = {}

    # Churn risk
    feature_cols = [
        "age", "sessions_per_week", "avg_session_minutes", "playtime_hours_total",
        "player_level", "achievements_unlocked", "matches_played", "win_ratio", "kd_ratio",
    ]
    X = df[feature_cols].copy()
    le_target = LabelEncoder()
    y = le_target.fit_transform(df["engagement_level"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    churn_model = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, min_samples_leaf=3
    )
    churn_model.fit(X_train, y_train)
    y_pred = churn_model.predict(X_test)
    clf_report = classification_report(
        y_test, y_pred, target_names=le_target.classes_, output_dict=True
    )
    importances = pd.Series(churn_model.feature_importances_, index=feature_cols).sort_values()

    low_idx = list(le_target.classes_).index("Low")
    df["quit_risk_score"] = churn_model.predict_proba(X)[:, low_idx]

    watchlist = (
        df[df["engagement_level"] == "Medium"]
        .sort_values("quit_risk_score", ascending=False)
        .head(15)
        [["player_id", "age", "game_genre", "sessions_per_week", "avg_session_minutes",
          "playtime_hours_total", "quit_risk_score"]]
    )
    n_flight_risk = len(df[(df.engagement_level == "Medium") & (df.quit_risk_score > 0.5)])
    n_medium = len(df[df.engagement_level == "Medium"])

    artifacts["churn"] = dict(
        clf_report=clf_report, importances=importances, watchlist=watchlist,
        n_flight_risk=n_flight_risk, n_medium=n_medium,
    )

    # Super fans / loyalty
    loyalty_features = [
        "sessions_per_week", "avg_session_minutes", "playtime_hours_total",
        "player_level", "achievements_unlocked", "matches_played",
    ]
    scaler_loyalty = StandardScaler()
    z = pd.DataFrame(
        scaler_loyalty.fit_transform(df[loyalty_features]),
        columns=loyalty_features, index=df.index,
    )
    df["loyalty_score"] = z.mean(axis=1)
    super_fans = df.sort_values("loyalty_score", ascending=False).head(20)
    loyalty_by_engagement = df.groupby("engagement_level")["loyalty_score"].mean().sort_values(ascending=False)

    artifacts["loyalty"] = dict(super_fans=super_fans, loyalty_by_engagement=loyalty_by_engagement,
                                 loyalty_features=loyalty_features)

    # Spend vs loyalty
    spend_vs_engagement = pd.crosstab(
        df["in_game_purchases"], df["engagement_level"], normalize="index"
    ) * 100
    spend_vs_engagement.index = spend_vs_engagement.index.map({0: "Non-payer", 1: "Payer"})

    payers = df.loc[df.in_game_purchases == 1, "playtime_hours_total"]
    nonpayers = df.loc[df.in_game_purchases == 0, "playtime_hours_total"]
    t_stat, p_val = stats.ttest_ind(payers, nonpayers, equal_var=False)
    r_purchase_playtime = df["purchase_amount_usd"].corr(df["playtime_hours_total"])
    r_purchase_sessions = df["purchase_amount_usd"].corr(df["sessions_per_week"])
    r_purchase_loyalty = df["purchase_amount_usd"].corr(df["loyalty_score"])
    verdict_spend = "confirms" if abs(r_purchase_loyalty) > 0.25 and p_val < 0.05 else "does NOT support"

    artifacts["spend"] = dict(
        spend_vs_engagement=spend_vs_engagement, payers=payers, nonpayers=nonpayers,
        t_stat=t_stat, p_val=p_val, r_purchase_playtime=r_purchase_playtime,
        r_purchase_sessions=r_purchase_sessions, r_purchase_loyalty=r_purchase_loyalty,
        verdict_spend=verdict_spend,
    )

    # Genre
    genre_summary = df.groupby("game_genre").agg(
        avg_playtime_hrs=("playtime_hours_total", "mean"),
        avg_sessions_per_wk=("sessions_per_week", "mean"),
        avg_session_min=("avg_session_minutes", "mean"),
        pct_high_engagement=("engagement_level", lambda s: (s == "High").mean() * 100),
        n_players=("player_id", "count"),
    ).sort_values("avg_playtime_hrs", ascending=False).round(1)
    genre_engagement = pd.crosstab(df["game_genre"], df["engagement_level"], normalize="index") * 100
    genre_engagement = genre_engagement[["Low", "Medium", "High"]].sort_values("High")
    best_genre = genre_summary["pct_high_engagement"].idxmax()
    worst_genre = genre_summary["pct_high_engagement"].idxmin()

    artifacts["genre"] = dict(
        genre_summary=genre_summary, genre_engagement=genre_engagement,
        best_genre=best_genre, worst_genre=worst_genre,
    )

    # Difficulty
    diff_order = [d for d in ["Easy", "Medium", "Hard"] if d in df["difficulty_preference"].unique()]
    diff_summary = df.groupby("difficulty_preference").agg(
        avg_playtime_hrs=("playtime_hours_total", "mean"),
        avg_sessions_per_wk=("sessions_per_week", "mean"),
        avg_win_ratio=("win_ratio", "mean"),
        pct_high_engagement=("engagement_level", lambda s: (s == "High").mean() * 100),
        n_players=("player_id", "count"),
    ).reindex(diff_order).round(2)
    diff_engagement = pd.crosstab(
        df["difficulty_preference"], df["engagement_level"], normalize="index"
    ).reindex(diff_order) * 100
    spread = diff_summary["pct_high_engagement"].max() - diff_summary["pct_high_engagement"].min()

    artifacts["difficulty"] = dict(
        diff_order=diff_order, diff_summary=diff_summary,
        diff_engagement=diff_engagement, spread=spread,
    )

    # Clustering
    cluster_features = [
        "sessions_per_week", "avg_session_minutes", "playtime_hours_total",
        "player_level", "win_ratio", "purchase_amount_usd",
    ]
    scaler_cluster = StandardScaler()
    Xc = scaler_cluster.fit_transform(df[cluster_features])

    sil_scores = {}
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(Xc)
        sil_scores[k] = silhouette_score(Xc, labels)

    best_k = max(sil_scores, key=sil_scores.get)
    K = best_k if sil_scores[best_k] > sil_scores.get(4, 0) + 0.05 else 4
    kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = kmeans.fit_predict(Xc)

    cluster_profile = df.groupby("cluster")[cluster_features + ["age"]].mean().round(1)
    cluster_profile["n_players"] = df["cluster"].value_counts().sort_index()

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(Xc)
    df["pca1"], df["pca2"] = pcs[:, 0], pcs[:, 1]
    explained_var = pca.explained_variance_ratio_.sum() * 100

    artifacts["cluster"] = dict(
        sil_scores=sil_scores, best_k=best_k, K=K, cluster_profile=cluster_profile,
        explained_var=explained_var,
    )

    # Prediction models
    predict_features_cat = ["gender", "game_genre", "difficulty_preference"]
    predict_features_num = ["age", "sessions_per_week", "avg_session_minutes"]
    Xp = pd.get_dummies(
        df[predict_features_num + predict_features_cat], columns=predict_features_cat, drop_first=True
    )

    y_playtime = df["playtime_hours_total"]
    Xtr, Xte, ytr, yte = train_test_split(Xp, y_playtime, test_size=0.25, random_state=RANDOM_STATE)
    playtime_model = RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, min_samples_leaf=3)
    playtime_model.fit(Xtr, ytr)
    pred_pt = playtime_model.predict(Xte)
    r2_playtime = r2_score(yte, pred_pt)
    mae_playtime = mean_absolute_error(yte, pred_pt)

    y_spend = df["purchase_amount_usd"]
    Xtr2, Xte2, ytr2, yte2 = train_test_split(Xp, y_spend, test_size=0.25, random_state=RANDOM_STATE)
    spend_model = RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, min_samples_leaf=3)
    spend_model.fit(Xtr2, ytr2)
    pred_sp = spend_model.predict(Xte2)
    r2_spend = r2_score(yte2, pred_sp)
    mae_spend = mean_absolute_error(yte2, pred_sp)

    pt_importance = pd.Series(playtime_model.feature_importances_, index=Xp.columns).sort_values()

    artifacts["predict"] = dict(
        playtime_model=playtime_model, spend_model=spend_model, Xp_columns=list(Xp.columns),
        predict_features_cat=predict_features_cat, predict_features_num=predict_features_num,
        yte=yte, pred_pt=pred_pt, yte2=yte2, pred_sp=pred_sp,
        r2_playtime=r2_playtime, mae_playtime=mae_playtime,
        r2_spend=r2_spend, mae_spend=mae_spend,
        pt_importance=pt_importance,
        gender_options=sorted(df["gender"].dropna().unique().tolist()),
        genre_options=sorted(df["game_genre"].dropna().unique().tolist()),
        difficulty_options=sorted(df["difficulty_preference"].dropna().unique().tolist()),
    )

    # Skill vs. time
    r_playtime_win = df["playtime_hours_total"].corr(df["win_ratio"])
    r_playtime_kd = df["playtime_hours_total"].corr(df["kd_ratio"])
    r_level_win = df["player_level"].corr(df["win_ratio"])

    df["playtime_bin"] = pd.qcut(df["playtime_hours_total"], q=10, duplicates="drop")
    skill_by_bin = df.groupby("playtime_bin", observed=True)[["win_ratio", "kd_ratio"]].mean()
    bin_midpoints = df.groupby("playtime_bin", observed=True)["playtime_hours_total"].mean()

    verdict_skill = (
        "playtime meaningfully predicts skill" if abs(r_playtime_win) > 0.25
        else "playtime does NOT meaningfully predict skill in this data"
    )

    artifacts["skill"] = dict(
        r_playtime_win=r_playtime_win, r_playtime_kd=r_playtime_kd, r_level_win=r_level_win,
        skill_by_bin=skill_by_bin, bin_midpoints=bin_midpoints, verdict_skill=verdict_skill,
    )

    return df, artifacts


df, art = run_pipeline(df_raw)

st.header("📊 Dashboard")

total_players = len(df)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total players", f"{total_players:,}")
k2.metric("Avg. playtime", f"{df['playtime_hours_total'].mean():.0f}h")
k3.metric("Avg. sessions/wk", f"{df['sessions_per_week'].mean():.1f}")
k4.metric("High engagement", f"{(df['engagement_level'] == 'High').mean() * 100:.1f}%")
k5.metric("Payer rate", f"{df['in_game_purchases'].mean() * 100:.1f}%")
k6.metric("Total revenue", f"${df['purchase_amount_usd'].sum():,.0f}")

st.markdown(
    f"🚩 **{art['churn']['n_flight_risk']}** Medium-engagement players are showing "
    "Low-engagement behavior (flight risk)."
)

dc1, dc2, dc3 = st.columns(3)

with dc1:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    counts = df["engagement_level"].value_counts().reindex(["Low", "Medium", "High"])
    ax.pie(counts, labels=counts.index, autopct="%1.0f%%",
           colors=sns.color_palette("viridis", 3), startangle=90)
    ax.set_title("Engagement level")
    plt.tight_layout()
    st.pyplot(fig)

with dc2:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    genre_counts = df["game_genre"].value_counts()
    genre_counts.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", len(genre_counts)))
    ax.set_title("Players by genre")
    ax.set_ylabel("Players")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

with dc3:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    seg_counts = df["cluster"].value_counts().sort_index()
    seg_counts.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", len(seg_counts)))
    ax.set_title("Players by segment")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Players")
    plt.tight_layout()
    st.pyplot(fig)

dc4, dc5 = st.columns(2)
with dc4:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["playtime_hours_total"], bins=30, ax=ax, color=sns.color_palette("viridis", 1)[0])
    ax.set_title("Playtime distribution")
    ax.set_xlabel("Total playtime (hours)")
    plt.tight_layout()
    st.pyplot(fig)

with dc5:
    fig, ax = plt.subplots(figsize=(6, 4))
    rev_by_engagement = df.groupby("engagement_level")["purchase_amount_usd"].sum().reindex(["Low", "Medium", "High"])
    rev_by_engagement.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", 3))
    ax.set_title("Revenue by engagement level")
    ax.set_ylabel("Total revenue ($)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

st.divider()
st.subheader("🔎 Explore in detail")

tabs = st.tabs([
    "📋 Overview", "🚩 Churn Risk", "⭐ Super Fans", "💸 Spend vs Loyalty",
    "🎯 Genre", "⚔️ Difficulty", "🧩 Segments", "🔮 Predict", "📈 Skill Curve", "📌 Summary",
])

# ---------------- Overview ----------------
with tabs[0]:
    st.subheader("Data preview")
    st.dataframe(df_raw.head(20), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Players", f"{len(df_raw):,}")
    c2.metric("Columns", df_raw.shape[1])
    c3.metric("Missing values", int(df_raw.isna().sum().sum()))
    st.subheader("Summary statistics")
    st.dataframe(df_raw.describe().T, use_container_width=True)

# ---------------- Churn ----------------
with tabs[1]:
    c = art["churn"]
    st.subheader("Churn-risk scoring")
    st.markdown("**Classification report** (held-out test set)")
    st.dataframe(pd.DataFrame(c["clf_report"]).T.round(3), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        c["importances"].plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(c["importances"])))
        ax.set_title("What predicts a player's engagement level?")
        ax.set_xlabel("Feature importance")
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=df, x="engagement_level", y="quit_risk_score",
                    order=["Low", "Medium", "High"], ax=ax)
        ax.set_title("Quit-risk score by current engagement level")
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("🚨 Flight-risk watchlist (top 15)")
    st.dataframe(c["watchlist"], use_container_width=True)

# ---------------- Super fans ----------------
with tabs[2]:
    l = art["loyalty"]
    st.subheader("⭐ Super fans")
    st.dataframe(
        l["super_fans"][["player_id", "age", "game_genre", "engagement_level"]
                        + l["loyalty_features"] + ["loyalty_score"]],
        use_container_width=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(data=df, x="engagement_level", y="loyalty_score",
                        order=["Low", "Medium", "High"], ax=ax)
        ax.set_title("Composite loyalty score by engagement level")
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        st.markdown("**Avg. loyalty score by engagement**")
        st.dataframe(l["loyalty_by_engagement"].rename("loyalty_score").round(3))

# ---------------- Spend vs loyalty ----------------
with tabs[3]:
    s = art["spend"]
    st.subheader("💸 Spend vs. loyalty")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(s["spend_vs_engagement"].round(1), use_container_width=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        s["spend_vs_engagement"][["Low", "Medium", "High"]].plot(
            kind="bar", stacked=True, ax=ax, color=sns.color_palette("viridis", 3)
        )
        ax.set_ylabel("% of players")
        ax.set_title("Engagement mix: payers vs. non-payers")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        sns.boxplot(data=df, x="in_game_purchases", y="playtime_hours_total", ax=axes[0])
        axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Non-payer", "Payer"])
        axes[0].set_title("Total playtime by purchase status")
        sns.scatterplot(data=df, x="purchase_amount_usd", y="loyalty_score",
                         hue="engagement_level", hue_order=["Low", "Medium", "High"],
                         alpha=0.5, ax=axes[1])
        axes[1].set_title("Spend ($) vs. loyalty score")
        plt.tight_layout()
        st.pyplot(fig)

# ---------------- Genre ----------------
with tabs[4]:
    g = art["genre"]
    st.subheader("🎯 Game genres")
    st.dataframe(g["genre_summary"], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        g["genre_summary"]["avg_playtime_hrs"].sort_values().plot(
            kind="barh", ax=ax, color=sns.color_palette("viridis", len(g["genre_summary"]))
        )
        ax.set_title("Avg. total playtime by genre")
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        g["genre_engagement"].plot(kind="barh", stacked=True, ax=ax, color=sns.color_palette("viridis", 3))
        ax.set_title("Engagement mix by genre (%)")
        plt.tight_layout()
        st.pyplot(fig)

# ---------------- Difficulty ----------------
with tabs[5]:
    d = art["difficulty"]
    st.subheader("⚔️ Difficulty preference")
    st.dataframe(d["diff_summary"], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=df, x="difficulty_preference", y="playtime_hours_total",
                    order=d["diff_order"], ax=ax, estimator=np.mean)
        ax.set_title("Avg. playtime by difficulty preference")
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        d["diff_engagement"][["Low", "Medium", "High"]].plot(
            kind="bar", stacked=True, ax=ax, color=sns.color_palette("viridis", 3)
        )
        ax.set_title("Engagement mix by difficulty (%)")
        plt.tight_layout()
        st.pyplot(fig)

# ---------------- Segmentation ----------------
with tabs[6]:
    cl = art["cluster"]
    st.subheader("🧩 Player segmentation (KMeans)")
    st.dataframe(cl["cluster_profile"], use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="pca1", y="pca2", hue="cluster", palette="viridis", alpha=0.6, ax=ax)
    ax.set_title(f"Player segments (PCA projection, {cl['explained_var']:.0f}% variance explained)")
    plt.tight_layout()
    st.pyplot(fig)

# ---------------- Prediction ----------------
with tabs[7]:
    p = art["predict"]
    st.subheader("🔮 Predictive models")
    m1, m2 = st.columns(2)
    m1.metric("Playtime model R²", f"{p['r2_playtime']:.3f}", help=f"MAE: {p['mae_playtime']:.1f} hours")
    m2.metric("Spend model R²", f"{p['r2_spend']:.3f}", help=f"MAE: ${p['mae_spend']:.2f}")

    st.subheader("🕹️ What-if predictor")
    with st.form("whatif_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=10, max_value=100, value=24)
            gender = st.selectbox("Gender", p["gender_options"])
        with c2:
            genre = st.selectbox("Game genre", p["genre_options"])
            difficulty = st.selectbox("Difficulty preference", p["difficulty_options"])
        with c3:
            sessions_per_week = st.number_input("Sessions per week", min_value=0, max_value=50, value=7)
            avg_session_minutes = st.number_input("Avg. session length (min)", min_value=0, max_value=600, value=35)
        submitted = st.form_submit_button("Predict")

    if submitted:
        row = pd.DataFrame([{
            "age": age, "sessions_per_week": sessions_per_week,
            "avg_session_minutes": avg_session_minutes,
            "gender": gender, "game_genre": genre, "difficulty_preference": difficulty,
        }])
        row_enc = pd.get_dummies(row, columns=p["predict_features_cat"], drop_first=True)
        row_enc = row_enc.reindex(columns=p["Xp_columns"], fill_value=0)
        playtime_pred = p["playtime_model"].predict(row_enc)[0]
        spend_pred = p["spend_model"].predict(row_enc)[0]

        r1, r2 = st.columns(2)
        r1.metric("Predicted total playtime", f"{playtime_pred:.1f} hours")
        r2.metric("Predicted spend", f"${spend_pred:.2f}")

# ---------------- Skill curve ----------------
with tabs[8]:
    sk = art["skill"]
    st.subheader("📈 Skill vs. time played")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(sk["bin_midpoints"], sk["skill_by_bin"]["win_ratio"], marker="o")
    axes[0].set_title("Win ratio across playtime deciles")
    axes[1].plot(sk["bin_midpoints"], sk["skill_by_bin"]["kd_ratio"], marker="o", color="darkorange")
    axes[1].set_title("K/D ratio across playtime deciles")
    plt.tight_layout()
    st.pyplot(fig)

# ---------------- Summary ----------------
with tabs[9]:
    st.subheader("📌 Summary findings")
    st.success("Analysis process completed successfully with standard data format!")