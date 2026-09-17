"""
🎮 Gaming Player Analytics — Streamlit App
Converted from the original Colab/Jupyter notebook.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
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

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Gaming Player Analytics", page_icon="🎮", layout="wide")
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.figsize"] = (9, 5)

st.title("🎮 Gaming Player Analytics")
st.caption(
    "Churn risk, super-fans, spend vs. loyalty, genre/difficulty retention, "
    "player segments, predictive models & skill curves."
)

with st.expander("ℹ️ About this dashboard / honesty note", expanded=False):
    st.markdown(
        """
This dataset is a **snapshot** (one row per player), not a time series — there's no
login-date or session-history field. So "who's about to quit" can't be measured directly
as *declining* activity. Instead, the churn section learns the behavioral signature of
players who are already `Low` engagement, then scores every other player against that
signature to build a **quit-risk score**. It's a solid, standard proxy — but it's worth
knowing that's what it is.
        """
    )

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
st.sidebar.header("📂 Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload gaming_player_stats.xlsx (or .csv)", type=["xlsx", "xls", "csv"]
)


@st.cache_data(show_spinner="Loading data...")
def load_data(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


if uploaded_file is None:
    st.info("👈 Upload your `gaming_player_stats.xlsx` file in the sidebar to get started.")
    st.stop()

df_uploaded = load_data(uploaded_file)

missing = [c for c in REQUIRED_COLS if c not in df_uploaded.columns]

# ----------------------------------------------------------------------
# Column mapping / format-conversion UI
# ----------------------------------------------------------------------
# Keeps a per-file mapping in session state so a re-run (or a fresh upload)
# doesn't carry over a stale mapping from a previously uploaded file.
mapping_key = f"col_mapping::{uploaded_file.name}::{uploaded_file.size}"


def _best_guess(required_col: str, available_cols: list) -> str:
    """Try to auto-suggest a source column for a required column by loose
    name matching, so the user isn't mapping every field from scratch."""
    norm = {c: c.strip().lower().replace(" ", "_").replace("-", "_") for c in available_cols}
    target = required_col.strip().lower()
    for col, n in norm.items():
        if n == target:
            return col
    for col, n in norm.items():
        if target in n or n in target:
            return col
    return "-- None --"


if missing:
    st.warning(
        "This file doesn't match our required format. Missing column(s): "
        + ", ".join(f"`{c}`" for c in missing)
        + ". You can map your file's columns to our expected format below instead of "
        "re-uploading a reformatted file."
    )

    with st.expander("🔄 Convert / map columns to our standard format", expanded=True):
        st.caption(
            "For each required field, choose the column from your file that corresponds to it. "
            "Fields already matching your file's headers are pre-filled."
        )
        available = list(df_uploaded.columns)
        options = ["-- None --"] + available

        if mapping_key not in st.session_state:
            st.session_state[mapping_key] = {
                rc: (rc if rc in available else _best_guess(rc, available))
                for rc in REQUIRED_COLS
            }

        cols_ui = st.columns(2)
        for i, req_col in enumerate(REQUIRED_COLS):
            current = st.session_state[mapping_key].get(req_col, "-- None --")
            idx = options.index(current) if current in options else 0
            with cols_ui[i % 2]:
                st.session_state[mapping_key][req_col] = st.selectbox(
                    f"`{req_col}`", options, index=idx, key=f"map_{mapping_key}_{req_col}"
                )

        apply_clicked = st.button("✅ Apply mapping & continue", type="primary")

    unmapped = [rc for rc, src in st.session_state[mapping_key].items() if src == "-- None --"]

    if not apply_clicked:
        if unmapped:
            st.info(f"Still unmapped: {', '.join(f'`{c}`' for c in unmapped)}")
        st.stop()

    if unmapped:
        st.error(
            "Please map every required field before continuing. Still unmapped: "
            + ", ".join(f"`{c}`" for c in unmapped)
        )
        st.stop()

    rename_map = {src: rc for rc, src in st.session_state[mapping_key].items()}
    df_raw = df_uploaded.rename(columns=rename_map)[REQUIRED_COLS + [
        c for c in df_uploaded.columns
        if c not in rename_map and c not in REQUIRED_COLS
    ]]
    st.success("Columns mapped to our standard format. Running the dashboard on your data below.")
else:
    df_raw = df_uploaded

st.sidebar.success(f"Loaded {len(df_raw):,} players, {df_raw.shape[1]} columns")

# ----------------------------------------------------------------------
# Cached analysis pipeline (mirrors the notebook, computed once per dataset)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Running analysis pipeline...")
def run_pipeline(df_in: pd.DataFrame):
    df = df_in.copy()
    artifacts = {}

    # ---------- 2. Churn risk ----------
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

    # ---------- 3. Super fans / loyalty ----------
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

    # ---------- 4. Spend vs loyalty ----------
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

    # ---------- 5. Genre ----------
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

    # ---------- 6. Difficulty ----------
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

    # ---------- 7. Clustering ----------
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

    def name_cluster(row, medians):
        tags = []
        tags.append("Frequent" if row["sessions_per_week"] >= medians["sessions_per_week"] else "Infrequent")
        tags.append("Long-session" if row["avg_session_minutes"] >= medians["avg_session_minutes"] else "Quick-session")
        tags.append("Big spender" if row["purchase_amount_usd"] >= medians["purchase_amount_usd"] else "Non-spender")
        tags.append("Skilled" if row["win_ratio"] >= medians["win_ratio"] else "Casual-skill")
        return " / ".join(tags)

    medians = df[cluster_features].median()
    cluster_profile["suggested_label"] = cluster_profile.apply(lambda r: name_cluster(r, medians), axis=1)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(Xc)
    df["pca1"], df["pca2"] = pcs[:, 0], pcs[:, 1]
    explained_var = pca.explained_variance_ratio_.sum() * 100

    artifacts["cluster"] = dict(
        sil_scores=sil_scores, best_k=best_k, K=K, cluster_profile=cluster_profile,
        explained_var=explained_var,
    )

    # ---------- 8. Prediction models ----------
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

    # ---------- 9. Skill vs. time ----------
    r_playtime_win = df["playtime_hours_total"].corr(df["win_ratio"])
    r_playtime_kd = df["playtime_hours_total"].corr(df["kd_ratio"])
    r_level_win = df["player_level"].corr(df["win_ratio"])

    df["playtime_bin"] = pd.qcut(df["playtime_hours_total"], q=10, duplicates="drop")
    skill_by_bin = df.groupby("playtime_bin", observed=True)[["win_ratio", "kd_ratio"]].mean()
    bin_midpoints = df.groupby("playtime_bin", observed=True)["playtime_hours_total"].mean()

    verdict_skill = (
        "playtime meaningfully predicts skill" if abs(r_playtime_win) > 0.25
        else "playtime does NOT meaningfully predict skill in this data — "
             "skill looks roughly flat regardless of hours played"
    )

    artifacts["skill"] = dict(
        r_playtime_win=r_playtime_win, r_playtime_kd=r_playtime_kd, r_level_win=r_level_win,
        skill_by_bin=skill_by_bin, bin_midpoints=bin_midpoints, verdict_skill=verdict_skill,
    )

    return df, artifacts


df, art = run_pipeline(df_raw)

# ----------------------------------------------------------------------
# Dashboard — shown immediately after upload, above the detailed tabs
# ----------------------------------------------------------------------
st.header("📊 Dashboard")

total_players = len(df)
avg_playtime = df["playtime_hours_total"].mean()
avg_sessions = df["sessions_per_week"].mean()
pct_high_engagement = (df["engagement_level"] == "High").mean() * 100
pct_payers = df["in_game_purchases"].mean() * 100
total_revenue = df["purchase_amount_usd"].sum()
flight_risk = art["churn"]["n_flight_risk"]

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total players", f"{total_players:,}")
k2.metric("Avg. playtime", f"{avg_playtime:.0f}h")
k3.metric("Avg. sessions/wk", f"{avg_sessions:.1f}")
k4.metric("High engagement", f"{pct_high_engagement:.1f}%")
k5.metric("Payer rate", f"{pct_payers:.1f}%")
k6.metric("Total revenue", f"${total_revenue:,.0f}")

st.markdown(f"🚩 **{flight_risk}** Medium-engagement players are showing Low-engagement behavior (flight risk).")

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
st.caption("Use the tabs below for the full breakdown — churn model, loyalty, genre/difficulty, segments, prediction, and skill analysis.")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
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
    st.markdown(
        "A Random Forest learns the behavioral fingerprint of `Low`-engagement players, "
        "then scores every player's probability of being one. Players currently `Medium` "
        "with a high score are your flight-risk watchlist."
    )
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

    st.markdown(
        f"**{c['n_flight_risk']} of {c['n_medium']}** Medium-engagement players score "
        f">0.5 quit-risk."
    )
    st.subheader("🚨 Flight-risk watchlist (top 15)")
    st.dataframe(c["watchlist"], use_container_width=True)

# ---------------- Super fans ----------------
with tabs[2]:
    l = art["loyalty"]
    st.subheader("⭐ Who are your super fans?")
    st.markdown(
        "Composite loyalty score built from standardized (z-scored) behavioral signals, "
        "so no single metric dominates."
    )
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
    st.subheader("💸 Does spending = loyalty?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Engagement mix (%) — payers vs non-payers**")
        st.dataframe(s["spend_vs_engagement"].round(1), use_container_width=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        s["spend_vs_engagement"][["Low", "Medium", "High"]].plot(
            kind="bar", stacked=True, ax=ax, color=sns.color_palette("viridis", 3)
        )
        ax.set_ylabel("% of players")
        ax.set_title("Engagement mix: payers vs. non-payers")
        ax.legend(title="Engagement")
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

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg playtime — payers", f"{s['payers'].mean():.1f}h")
    m2.metric("Avg playtime — non-payers", f"{s['nonpayers'].mean():.1f}h")
    m3.metric("t-test p-value", f"{s['p_val']:.3f}")
    m4.metric("r(spend, loyalty)", f"{s['r_purchase_loyalty']:+.3f}")
    st.markdown(
        f"Correlation, $ spent vs. total playtime: **r={s['r_purchase_playtime']:+.3f}** · "
        f"$ spent vs. sessions/week: **r={s['r_purchase_sessions']:+.3f}**"
    )
    st.info(f"This dataset **{s['verdict_spend']}** the 'spenders = most engaged' assumption.")

# ---------------- Genre ----------------
with tabs[4]:
    g = art["genre"]
    st.subheader("🎯 Which game genres keep players hooked?")
    st.dataframe(g["genre_summary"], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        g["genre_summary"]["avg_playtime_hrs"].sort_values().plot(
            kind="barh", ax=ax, color=sns.color_palette("viridis", len(g["genre_summary"]))
        )
        ax.set_title("Avg. total playtime by genre")
        ax.set_xlabel("Hours")
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        g["genre_engagement"].plot(kind="barh", stacked=True, ax=ax, color=sns.color_palette("viridis", 3))
        ax.set_title("Engagement mix by genre (%)")
        ax.legend(title="Engagement", bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        st.pyplot(fig)

    st.success(
        f"**Stickiest genre:** {g['best_genre']} "
        f"({g['genre_summary'].loc[g['best_genre'], 'pct_high_engagement']:.1f}% High-engagement)  \n"
        f"**Least sticky genre:** {g['worst_genre']} "
        f"({g['genre_summary'].loc[g['worst_genre'], 'pct_high_engagement']:.1f}% High-engagement)"
    )

# ---------------- Difficulty ----------------
with tabs[5]:
    d = art["difficulty"]
    st.subheader("⚔️ Does difficulty drive people away or hook them in?")
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
        ax.set_xticklabels(d["diff_order"], rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown(
        f"Spread in High-engagement % across difficulty levels: **{d['spread']:.1f} points**  \n"
        "A small spread means difficulty isn't doing much to drive people away or hook them; "
        "a large spread means it matters a lot."
    )

# ---------------- Segmentation ----------------
with tabs[6]:
    cl = art["cluster"]
    st.subheader("🧩 Player segmentation (KMeans)")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(cl["sil_scores"].keys()), list(cl["sil_scores"].values()), marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Choosing k")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown(
        f"Best silhouette score at **k={cl['best_k']}** "
        f"({cl['sil_scores'][cl['best_k']]:.3f}). Using **K={cl['K']}** segments."
    )

    st.markdown("**Cluster profiles**")
    st.dataframe(cl["cluster_profile"], use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="pca1", y="pca2", hue="cluster", palette="viridis", alpha=0.6, ax=ax)
    ax.set_title(f"Player segments (PCA projection, {cl['explained_var']:.0f}% variance explained)")
    plt.tight_layout()
    st.pyplot(fig)

# ---------------- Prediction ----------------
with tabs[7]:
    p = art["predict"]
    st.subheader("🔮 Predicting playtime & spend")
    m1, m2 = st.columns(2)
    m1.metric("Playtime model R²", f"{p['r2_playtime']:.3f}", help=f"MAE: {p['mae_playtime']:.1f} hours")
    m2.metric("Spend model R²", f"{p['r2_spend']:.3f}", help=f"MAE: ${p['mae_spend']:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.scatter(p["yte"], p["pred_pt"], alpha=0.4)
        ax.plot([p["yte"].min(), p["yte"].max()], [p["yte"].min(), p["yte"].max()], "r--")
        ax.set_xlabel("Actual playtime (hrs)"); ax.set_ylabel("Predicted playtime (hrs)")
        ax.set_title(f"Playtime: predicted vs. actual (R²={p['r2_playtime']:.2f})")
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.scatter(p["yte2"], p["pred_sp"], alpha=0.4, color="darkorange")
        ax.plot([p["yte2"].min(), p["yte2"].max()], [p["yte2"].min(), p["yte2"].max()], "r--")
        ax.set_xlabel("Actual spend ($)"); ax.set_ylabel("Predicted spend ($)")
        ax.set_title(f"Spend: predicted vs. actual (R²={p['r2_spend']:.2f})")
        plt.tight_layout()
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    p["pt_importance"].plot(kind="barh", ax=ax)
    ax.set_title("What drives predicted playtime?")
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()
    st.subheader("🕹️ What-if predictor")
    st.markdown("Try a hypothetical player and see predicted total playtime & spend.")

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
    st.subheader("📈 Skill vs. time played — do people get better, or plateau?")
    m1, m2, m3 = st.columns(3)
    m1.metric("r(playtime, win_ratio)", f"{sk['r_playtime_win']:+.3f}")
    m2.metric("r(playtime, kd_ratio)", f"{sk['r_playtime_kd']:+.3f}")
    m3.metric("r(level, win_ratio)", f"{sk['r_level_win']:+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(sk["bin_midpoints"], sk["skill_by_bin"]["win_ratio"], marker="o")
    axes[0].set_xlabel("Avg. playtime in bin (hrs)"); axes[0].set_ylabel("Mean win ratio")
    axes[0].set_title("Win ratio across playtime deciles")
    axes[1].plot(sk["bin_midpoints"], sk["skill_by_bin"]["kd_ratio"], marker="o", color="darkorange")
    axes[1].set_xlabel("Avg. playtime in bin (hrs)"); axes[1].set_ylabel("Mean K/D ratio")
    axes[1].set_title("K/D ratio across playtime deciles")
    plt.tight_layout()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if HAS_STATSMODELS:
        sns.regplot(data=df, x="playtime_hours_total", y="win_ratio", scatter_kws={"alpha": 0.25},
                    lowess=True, line_kws={"color": "red"}, ax=ax)
        ax.set_title("Win ratio vs. total playtime (with LOWESS trend)")
    else:
        sns.regplot(data=df, x="playtime_hours_total", y="win_ratio", scatter_kws={"alpha": 0.25},
                    line_kws={"color": "red"}, ax=ax)
        ax.set_title("Win ratio vs. total playtime (linear trend — install statsmodels for LOWESS)")
    plt.tight_layout()
    st.pyplot(fig)

    st.info(f"**Verdict:** {sk['verdict_skill']}.")

# ---------------- Summary ----------------
with tabs[9]:
    c, l, s, g, d, cl, p, sk = (
        art["churn"], art["loyalty"], art["spend"], art["genre"],
        art["difficulty"], art["cluster"], art["predict"], art["skill"],
    )
    st.subheader("📌 Key findings")
    st.markdown(f"""
**1. Churn risk**
- Behavioral model separates engagement tiers with real signal (see classification report).
- **{c['n_flight_risk']} of {c['n_medium']}** Medium-engagement players already look behaviorally
  like Low-engagement players — that's your win-back list.

**2. Super fans**
- Top loyalty-score players average **{l['super_fans']['sessions_per_week'].mean():.1f}** sessions/week
  and **{l['super_fans']['playtime_hours_total'].mean():.0f}** total hours, vs.
  **{df['sessions_per_week'].mean():.1f}** and **{df['playtime_hours_total'].mean():.0f}** for the full base.

**3. Spend vs. loyalty**
- Correlation between $ spent and loyalty score: **r={s['r_purchase_loyalty']:+.3f}**
- Payers vs non-payers playtime: **{s['payers'].mean():.1f}h** vs **{s['nonpayers'].mean():.1f}h**
  (p={s['p_val']:.3f})
- Verdict: this dataset **{s['verdict_spend']}** the "spenders = most engaged" assumption.

**4. Genre**
- Stickiest genre: **{g['best_genre']}** · Least sticky: **{g['worst_genre']}**

**5. Difficulty**
- Spread in High-engagement % across difficulty levels: **{d['spread']:.1f} points**
  (small = difficulty barely matters here; large = it matters a lot)

**6. Segmentation**
- **{cl['K']}** behavioral segments identified via KMeans
  (silhouette={cl['sil_scores'].get(cl['K'], cl['sil_scores'][cl['best_k']]):.3f})

**7. Prediction**
- Playtime model R²={p['r2_playtime']:.3f}, MAE={p['mae_playtime']:.1f}h
- Spend model R²={p['r2_spend']:.3f}, MAE=${p['mae_spend']:.2f}

**8. Skill vs. time played**
- r(playtime, win_ratio)={sk['r_playtime_win']:+.3f}, r(playtime, kd_ratio)={sk['r_playtime_kd']:+.3f}
- Verdict: {sk['verdict_skill']}
""")