import os
import re
import json

from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import seaborn as sns
import networkx as nx

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def find_chinese_font() -> Optional[str]:
    font_candidates = [
        ('C:/Windows/Fonts/msyh.ttc', 'Microsoft YaHei'),
        ('C:/Windows/Fonts/msyhbd.ttc', 'Microsoft YaHei'),
        ('C:/Windows/Fonts/simhei.ttf', 'SimHei'),
        ('C:/Windows/Fonts/simsun.ttc', 'SimSun'),
    ]
    for font_path, font_name in font_candidates:
        if os.path.exists(font_path):
            return font_path
    return None

def setup_chinese_font():
    font_path = find_chinese_font()
    if font_path:
        try:
            fm._load_fontmanager(try_read_cache=False)
        except:
            pass
        font_prop = fm.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial']
        mpl.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
        sns.set_context("paper", rc={"font.family": "sans-serif", "font.sans-serif": [font_name]})
        return font_path
    else:
        mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        mpl.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
        return None

CHINESE_FONT_PATH = setup_chinese_font()

def get_font_properties():
    if CHINESE_FONT_PATH:
        return fm.FontProperties(fname=CHINESE_FONT_PATH)
    return fm.FontProperties()

MAJOR_EVENTS = [
    {
        "name_cn": "巴黎恐怖袭击事件",
        "name_en": "Paris Terror Attacks",
        "start_date": "2015-11-13",
        "end_date": "2015-11-20",
        "peak_date": "2015-11-14",
        "keywords_fr": ["attentat", "attaque", "terroriste", "bataclan", "stade", "france", 
                        "mort", "victime", "daesh", "isis", "explosion", "fusillade"],
        "keywords_en": ["attack", "terror", "terrorist", "bataclan", "stadium", "dead", 
                        "victim", "isis", "explosion", "shooting", "hostage", "police"],
        "keywords_cn": ["恐怖", "袭击", "巴黎", "法国", "爆炸", "枪击", "死亡", "哀悼"],
        "location": "Paris",
        "severity": 10
    },
    {
        "name_cn": "法国地区选举",
        "name_en": "French Regional Elections",
        "start_date": "2015-11-01",
        "end_date": "2015-11-15",
        "peak_date": "2015-11-06",
        "keywords_fr": ["election", "vote", "scrutin", "candidat", "campagne", "sondage"],
        "keywords_en": ["election", "vote", "candidate", "campaign", "poll", "political"],
        "keywords_cn": ["选举", "投票", "候选人"],
        "location": "France",
        "severity": 6
    },
    {
        "name_cn": "COP21气候大会预热",
        "name_en": "COP21 Climate Summit Preparation",
        "start_date": "2015-11-20",
        "end_date": "2015-11-30",
        "peak_date": "2015-11-28",
        "keywords_fr": ["cop21", "climat", "environnement", "rechauffement", "carbone"],
        "keywords_en": ["cop21", "climate", "environment", "warming", "carbon", "summit"],
        "keywords_cn": ["气候", "环境", "大会"],
        "location": "Paris",
        "severity": 7
    },
    {
        "name_cn": "法国足球赛事",
        "name_en": "French Football Matches",
        "start_date": "2015-11-01",
        "end_date": "2015-11-30",
        "peak_date": "2015-11-15",
        "keywords_fr": ["match", "football", "foot", "equipe", "but", "stade", "psg", "victoire"],
        "keywords_en": ["match", "football", "soccer", "goal", "team", "stadium", "win"],
        "keywords_cn": ["足球", "比赛", "球队"],
        "location": "France",
        "severity": 4
    },
    {
        "name_cn": "万圣节/诸圣节",
        "name_en": "All Saints Day / Halloween",
        "start_date": "2015-10-31",
        "end_date": "2015-11-02",
        "peak_date": "2015-11-01",
        "keywords_fr": ["toussaint", "halloween", "vacances", "ferie", "famille"],
        "keywords_en": ["halloween", "holiday", "vacation", "family"],
        "keywords_cn": ["万圣节", "假期"],
        "location": "France",
        "severity": 3
    }
]

TOPIC_KEYWORDS = {
    "社会议题": {
        "keywords_fr": ["politique", "gouvernement", "president", "ministre", "loi"],
        "keywords_en": ["politics", "government", "president", "minister", "law"],
        "keywords_cn": ["政治", "政府", "改革", "社会", "抗议"]
    },
    "娱乐时尚": {
        "keywords_fr": ["film", "musique", "concert", "star", "mode"],
        "keywords_en": ["movie", "music", "concert", "star", "fashion"],
        "keywords_cn": ["电影", "音乐", "明星", "时尚", "娱乐"]
    },
    "体育赛事": {
        "keywords_fr": ["football", "foot", "match", "equipe", "victoire"],
        "keywords_en": ["football", "soccer", "match", "team", "goal"],
        "keywords_cn": ["足球", "比赛", "运动", "冠军"]
    }
}

def ensure_vader_lexicon() -> None:
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

def load_tweet_data(csv_path: str | None = None) -> pd.DataFrame:
    if csv_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "csv", "output3.csv")
    df = pd.read_csv(csv_path)
    for col in ["favorites_count", "user_followers_count", "user_statuses_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "tweet_content" not in df.columns:
        raise KeyError("CSV 中未找到 'tweet_content' 列")
    df["tweet_length"] = df["tweet_content"].astype(str).str.len()
    ensure_vader_lexicon()
    sia = SentimentIntensityAnalyzer()

    def compute_emotion_index(text: str) -> float:
        scores = sia.polarity_scores(str(text))
        compound = scores.get("compound", 0.0)
        return (compound + 1.0) / 2.0

    df["emotion_index"] = df["tweet_content"].astype(str).map(compute_emotion_index)
    if "tweet_posted_time" in df.columns:
        df["posted_datetime"] = pd.to_datetime(df["tweet_posted_time"], errors="coerce")
        df["posted_hour"] = df["posted_datetime"].dt.hour
    if "favorites_count" in df.columns:
        median_fav = df["favorites_count"].median()
        df["is_high_favorite"] = df["favorites_count"] >= median_fav
    return df

def calculate_event_relevance_vectorized(texts: pd.Series, event: Dict, dates: pd.Series = None) -> pd.Series:
    all_keywords = (
        event.get("keywords_fr", []) + 
        event.get("keywords_en", []) + 
        event.get("keywords_cn", [])
    )
    if not all_keywords:
        return pd.Series(0.0, index=texts.index)
    texts_lower = texts.astype(str).str.lower()
    total_weight = sum(len(kw) / 5.0 for kw in all_keywords)
    matched_weights = pd.Series(0.0, index=texts.index)
    for keyword in all_keywords:
        kw_lower = keyword.lower()
        weight = len(kw_lower) / 5.0
        matched_weights += texts_lower.str.contains(kw_lower, regex=False, na=False).astype(float) * weight
    keyword_score = matched_weights / max(total_weight, 1)
    time_score = pd.Series(0.5, index=texts.index)
    if dates is not None:
        try:
            start_date = pd.to_datetime(event["start_date"])
            end_date = pd.to_datetime(event["end_date"])
            peak_date = pd.to_datetime(event["peak_date"])
            in_range = (dates >= start_date) & (dates <= end_date)
            days_from_peak = (dates - peak_date).dt.days.abs()
            time_score = pd.Series(0.3, index=texts.index)
            time_score[in_range] = (1.0 - days_from_peak[in_range] * 0.1).clip(lower=0.5)
        except:
            pass
    relevance = keyword_score * 0.7 + time_score * 0.3
    severity_factor = event.get("severity", 5) / 10.0
    relevance = relevance * (0.5 + severity_factor * 0.5)
    return relevance.clip(upper=1.0)

def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "tweet_posted_time" in df.columns:
        df["posted_datetime"] = pd.to_datetime(df["tweet_posted_time"], errors="coerce")
        df["posted_date"] = df["posted_datetime"].dt.date
    texts = df["tweet_content"].astype(str)
    dates = df.get("posted_datetime", None)
    event_cols = []
    for i, event in enumerate(MAJOR_EVENTS):
        event_name = event["name_cn"].replace(" ", "_")
        col_name = f"event_relevance_{event_name}"
        df[col_name] = calculate_event_relevance_vectorized(texts, event, dates)
        event_cols.append(col_name)
    relevance_df = df[event_cols]
    df["max_event_relevance"] = relevance_df.max(axis=1)
    max_idx = relevance_df.idxmax(axis=1)
    col_to_event = {f"event_relevance_{e['name_cn'].replace(' ', '_')}": e["name_cn"] for e in MAJOR_EVENTS}
    df["primary_event"] = max_idx.map(col_to_event)
    df.loc[df["max_event_relevance"] < 0.1, "primary_event"] = "无关联事件"
    return df

def plot_event_histogram(df: pd.DataFrame) -> None:
    font_prop = get_font_properties()
    if "primary_event" not in df.columns:
        return
    event_counts = df["primary_event"].value_counts()
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(event_counts)))
    bars = ax.bar(range(len(event_counts)), event_counts.values, color=colors, edgecolor="black")
    ax.set_xticks(range(len(event_counts)))
    ax.set_xticklabels(event_counts.index, rotation=45, ha="right", fontproperties=font_prop)
    ax.set_xlabel("重大事件", fontproperties=font_prop)
    ax.set_ylabel("推文数量", fontproperties=font_prop)
    ax.set_title("推文中包含的当时当地重大事件统计", fontproperties=font_prop)
    for bar, count in zip(bars, event_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                 str(count), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig("event_histogram.png", dpi=300)
    plt.show()

def analyze_event_relevance_distribution(df: pd.DataFrame) -> None:
    font_prop = get_font_properties()
    if "max_event_relevance" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(df["max_event_relevance"].dropna(), bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(df["max_event_relevance"].mean(), color="red", linestyle="--", 
               label=f'平均值: {df["max_event_relevance"].mean():.3f}')
    ax.axvline(df["max_event_relevance"].median(), color="orange", linestyle="--", 
               label=f'中位数: {df["max_event_relevance"].median():.3f}')
    ax.set_xlabel("事件关联度", fontproperties=font_prop)
    ax.set_ylabel("推文数量", fontproperties=font_prop)
    ax.set_title("推文与重大事件关联度分布", fontproperties=font_prop)
    ax.legend(prop=font_prop)
    event_cols = [f"event_relevance_{e['name_cn'].replace(' ', '_')}" for e in MAJOR_EVENTS]
    event_names = [e["name_cn"] for e in MAJOR_EVENTS]
    violin_data = []
    for col, name in zip(event_cols, event_names):
        if col in df.columns:
            data = df[col].dropna()
            for val in data:
                violin_data.append({"event": name, "relevance": val})
    ax = axes[1]
    violin_df = pd.DataFrame(violin_data)
    if not violin_df.empty:
        sns.violinplot(data=violin_df, x="event", y="relevance", ax=ax, palette="Set2")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontproperties=font_prop)
        ax.set_xlabel("重大事件", fontproperties=font_prop)
        ax.set_ylabel("关联度", fontproperties=font_prop)
        ax.set_title("各重大事件关联度分布", fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig("event_relevance_distribution.png", dpi=300)
    plt.show()

def plot_event_relevance_over_time(df: pd.DataFrame) -> None:
    font_prop = get_font_properties()
    if "posted_date" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, len(MAJOR_EVENTS)))
    for i, event in enumerate(MAJOR_EVENTS):
        event_name = event["name_cn"]
        col_name = f"event_relevance_{event_name.replace(' ', '_')}"
        if col_name in df.columns:
            daily_relevance = df.groupby("posted_date")[col_name].mean()
            dates = [pd.to_datetime(d) for d in daily_relevance.index]
            ax.plot(dates, daily_relevance.values, color=colors[i], linewidth=2, 
                   label=event_name, marker="o", markersize=3)
    ax.set_xlabel("日期", fontproperties=font_prop)
    ax.set_ylabel("平均关联度", fontproperties=font_prop)
    ax.set_title("各重大事件关联度随时间变化对比", fontproperties=font_prop)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", prop=font_prop)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("event_relevance_over_time.png", dpi=300)
    plt.show()

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "favorites_count" in df.columns and "user_statuses_count" in df.columns:
        df["favorites_count"] = pd.to_numeric(df["favorites_count"], errors="coerce")
        df["user_statuses_count"] = pd.to_numeric(df["user_statuses_count"], errors="coerce")
        df["fav_status_ratio"] = df["favorites_count"] / df["user_statuses_count"].replace(0, np.nan)
    if "in_reply_to_status_id" in df.columns:
        df["is_reply"] = df["in_reply_to_status_id"].notna().astype(int)
    else:
        df["is_reply"] = 0
    if "in_reply_to_status_id" in df.columns and "tweet_id" in df.columns:
        reply_counts = df["in_reply_to_status_id"].value_counts()
        df["reply_received"] = df["tweet_id"].map(reply_counts).fillna(0).astype(int)
    else:
        df["reply_received"] = 0
    df["total_reply_activity"] = df["is_reply"] + df["reply_received"]
    if "user_followers_count" in df.columns:
        median_followers = df["user_followers_count"].median()
        df["is_high_follower"] = df["user_followers_count"] >= median_followers
        df["user_type"] = df["is_high_follower"].map({True: "高粉丝用户", False: "普通用户"})
    return df

def plot_quadrant_scatter(x, y, xlabel: str, ylabel: str, title: str, filename: str, center_origin: bool = False) -> None:
    font_prop = get_font_properties()
    mask = pd.notna(x) & pd.notna(y) & np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]
    if len(x_valid) < 10:
        return
    x_median = x_valid.median()
    y_median = y_valid.median()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x_valid, y_valid, alpha=0.5, c="steelblue", s=30)
    ax.axhline(y=y_median, color="red", linestyle="--", alpha=0.7)
    ax.axvline(x=x_median, color="red", linestyle="--", alpha=0.7)
    q1 = ((x_valid > x_median) & (y_valid > y_median)).sum()
    q2 = ((x_valid <= x_median) & (y_valid > y_median)).sum()
    q3 = ((x_valid <= x_median) & (y_valid <= y_median)).sum()
    q4 = ((x_valid > x_median) & (y_valid <= y_median)).sum()
    if center_origin:
        x_max_dist = max(abs(x_valid.max() - x_median), abs(x_valid.min() - x_median))
        y_max_dist = max(abs(y_valid.max() - y_median), abs(y_valid.min() - y_median))
        x_max_dist *= 1.1
        y_max_dist *= 1.1
        ax.set_xlim(x_median - x_max_dist, x_median + x_max_dist)
        ax.set_ylim(y_median - y_max_dist, y_median + y_max_dist)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    ax.text(x_median + x_range*0.25, y_median + y_range*0.25, f"I象限\n(n={q1})", 
             fontsize=12, ha="center", color="darkgreen", fontweight="bold", fontproperties=font_prop)
    ax.text(x_median - x_range*0.25, y_median + y_range*0.25, f"II象限\n(n={q2})", 
             fontsize=12, ha="center", color="darkblue", fontweight="bold", fontproperties=font_prop)
    ax.text(x_median - x_range*0.25, y_median - y_range*0.25, f"III象限\n(n={q3})", 
             fontsize=12, ha="center", color="darkred", fontweight="bold", fontproperties=font_prop)
    ax.text(x_median + x_range*0.25, y_median - y_range*0.25, f"IV象限\n(n={q4})", 
             fontsize=12, ha="center", color="darkorange", fontweight="bold", fontproperties=font_prop)
    ax.set_xlabel(xlabel, fontproperties=font_prop)
    ax.set_ylabel(ylabel, fontproperties=font_prop)
    ax.set_title(title, fontproperties=font_prop)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_radar_chart(df: pd.DataFrame) -> None:
    font_prop = get_font_properties()
    if "user_type" not in df.columns:
        return
    metrics = []
    if "emotion_index" in df.columns:
        metrics.append(("emotion_index", "情绪指数"))
    if "max_event_relevance" in df.columns:
        metrics.append(("max_event_relevance", "事件关联度"))
    if "tweet_length" in df.columns:
        metrics.append(("tweet_length", "推文长度"))
    if "total_reply_activity" in df.columns:
        metrics.append(("total_reply_activity", "回复活跃度"))
    if "favorites_count" in df.columns:
        metrics.append(("favorites_count", "收藏量"))
    if len(metrics) < 3:
        return
    high_user = df[df["user_type"] == "高粉丝用户"]
    normal_user = df[df["user_type"] == "普通用户"]
    high_values = []
    normal_values = []
    labels = []
    for col, label in metrics:
        h_mean = high_user[col].mean()
        n_mean = normal_user[col].mean()
        max_val = max(h_mean, n_mean, 1)
        high_values.append(h_mean / max_val)
        normal_values.append(n_mean / max_val)
        labels.append(label)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    high_values += high_values[:1]
    normal_values += normal_values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, high_values, 'o-', linewidth=2, label="高粉丝用户", color="blue")
    ax.fill(angles, high_values, alpha=0.25, color="blue")
    ax.plot(angles, normal_values, 'o-', linewidth=2, label="普通用户", color="green")
    ax.fill(angles, normal_values, alpha=0.25, color="green")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontproperties=font_prop)
    ax.set_title("高粉丝用户 vs 普通用户对比雷达图", fontproperties=font_prop, fontsize=14)
    ax.legend(loc="upper right", prop=font_prop)
    plt.tight_layout()
    plt.savefig("user_type_radar.png", dpi=300)
    plt.show()

def plot_reply_network_force_directed(df: pd.DataFrame, event_name: str = None) -> None:
    font_prop = get_font_properties()
    required_cols = ["user_name", "in_reply_to_screen_name", "user_followers_count"]
    if not all(col in df.columns for col in required_cols):
        return
    data = df[df["primary_event"] == event_name] if event_name and "primary_event" in df.columns else df
    title_suffix = f" - {event_name}" if event_name else ""
    reply_data = data[data["in_reply_to_screen_name"].notna()][["user_name", "in_reply_to_screen_name", "user_followers_count"]].copy()
    if len(reply_data) < 10:
        return
    reply_counts = reply_data["user_name"].value_counts().head(30)
    replied_counts = reply_data["in_reply_to_screen_name"].value_counts().head(30)
    top_users = set(reply_counts.index) | set(replied_counts.index)
    mask = reply_data["user_name"].isin(top_users) | reply_data["in_reply_to_screen_name"].isin(top_users)
    reply_data = reply_data[mask]
    reply_data["source"] = reply_data["user_name"].astype(str).str[:15]
    reply_data["target"] = reply_data["in_reply_to_screen_name"].astype(str).str[:15]
    reply_data = reply_data[reply_data["source"] != reply_data["target"]]
    edge_counts = reply_data.groupby(["source", "target"]).size().reset_index(name="weight")
    if len(edge_counts) < 2:
        return
    G = nx.DiGraph()
    for _, row in edge_counts.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight"])
    if G.number_of_nodes() < 2:
        return
    in_degree = dict(G.in_degree())
    user_followers = data.groupby("user_name")["user_followers_count"].first().to_dict()
    node_influence = {}
    for node in G.nodes():
        followers = user_followers.get(node, 0)
        if pd.isna(followers):
            followers = 0
        node_influence[node] = in_degree.get(node, 0) * 0.5 + followers / 10000
    max_influence = max(node_influence.values()) if node_influence else 1
    normalized_influence = {k: v / max_influence for k, v in node_influence.items()}
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.random_layout(G, seed=42)
    node_sizes = [300 + node_influence.get(n, 0) * 100 for n in G.nodes()]
    node_colors = [normalized_influence.get(n, 0) for n in G.nodes()]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + w / max_weight * 2 for w in edge_weights]
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, 
                          width=edge_widths, arrows=True, arrowsize=8, ax=ax)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                   node_color=node_colors, cmap=plt.cm.RdYlGn_r,
                                   alpha=0.8, edgecolors='black', linewidths=0.5, ax=ax)
    top_nodes = sorted(node_influence.items(), key=lambda x: x[1], reverse=True)[:10]
    top_node_names = {n[0] for n in top_nodes}
    labels = {n: n for n in G.nodes() if n in top_node_names}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    cbar = plt.colorbar(nodes, ax=ax)
    cbar.set_label("影响力", fontproperties=font_prop)
    ax.set_title(f"回复网络力导向图{title_suffix}\n(节点大小和颜色按影响力排序)", 
                fontsize=14, fontproperties=font_prop)
    ax.axis('off')
    plt.tight_layout()
    filename = f"reply_network_force{'_' + event_name.replace(' ', '_') if event_name else ''}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_event_propagation_tree(df: pd.DataFrame) -> None:
    font_prop = get_font_properties()
    if "primary_event" not in df.columns:
        return
    required_cols = ["tweet_id", "in_reply_to_status_id", "user_name", "favorites_count"]
    if not all(col in df.columns for col in required_cols):
        return
    events = [e for e in df["primary_event"].unique() if e != "无关联事件"]
    if len(events) == 0:
        return
    n_events = min(len(events), 4)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    for i, event in enumerate(events[:n_events]):
        ax = axes[i]
        event_data = df[df["primary_event"] == event].copy()
        G = nx.DiGraph()
        reply_data = event_data[event_data["in_reply_to_status_id"].notna()]
        if len(reply_data) < 5:
            ax.text(0.5, 0.5, f"{event}\n数据不足", ha='center', va='center', 
                   fontsize=12, fontproperties=font_prop)
            ax.axis('off')
            continue
        tweet_to_user = dict(zip(event_data["tweet_id"].astype(str), event_data["user_name"]))
        tweet_to_fav = dict(zip(event_data["tweet_id"].astype(str), event_data["favorites_count"]))
        for _, row in reply_data.head(100).iterrows():
            child_id = str(row["tweet_id"])
            parent_id = str(row["in_reply_to_status_id"])
            child_user = str(row["user_name"])[:12]
            parent_user = tweet_to_user.get(parent_id, "unknown")[:12]
            if child_user != parent_user:
                G.add_node(child_user, fav=row.get("favorites_count", 0))
                G.add_node(parent_user, fav=tweet_to_fav.get(parent_id, 0))
                G.add_edge(parent_user, child_user)
        if G.number_of_nodes() < 2:
            ax.text(0.5, 0.5, f"{event}\n数据不足", ha='center', va='center', 
                   fontsize=12, fontproperties=font_prop)
            ax.axis('off')
            continue
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.random_layout(G, seed=42)
        out_degrees = dict(G.out_degree())
        node_sizes = [200 + out_degrees.get(n, 0) * 50 for n in G.nodes()]
        in_degrees = dict(G.in_degree())
        node_colors = [1 - in_degrees.get(n, 0) / max(max(in_degrees.values()), 1) for n in G.nodes()]
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.5, 
                              arrows=True, arrowsize=8, width=0.5)
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, 
                                       node_color=node_colors, cmap=plt.cm.Oranges,
                                       alpha=0.8, edgecolors='black', linewidths=0.3)
        core_nodes = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:8]
        labels = {n[0]: n[0] for n in core_nodes}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7)
        ax.set_title(f"{event}\n(节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()})", 
                    fontsize=11, fontproperties=font_prop)
        ax.axis('off')
    for j in range(n_events, 4):
        axes[j].axis('off')
    plt.suptitle("不同重大事件的舆论传播树状图", fontsize=14, fontweight='bold', fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig("event_propagation_tree.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_event_emotion_pie_charts(df: pd.DataFrame) -> None:
    font_prop = get_font_properties()
    if "primary_event" not in df.columns or "emotion_index" not in df.columns:
        return
    events = [e for e in df["primary_event"].unique() if e != "无关联事件"][:4]
    if len(events) == 0:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    for i, event in enumerate(events):
        ax = axes[i]
        event_data = df[df["primary_event"] == event]["emotion_index"]
        positive = (event_data > 0.6).sum()
        neutral = ((event_data >= 0.4) & (event_data <= 0.6)).sum()
        negative = (event_data < 0.4).sum()
        sizes = [positive, neutral, negative]
        labels = ['积极', '中性', '消极']
        colors = ['#66b3ff', '#99ff99', '#ff9999']
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
            textprops={'fontproperties': font_prop}
        )
        for text in texts:
            text.set_fontproperties(font_prop)
        for autotext in autotexts:
            autotext.set_fontproperties(font_prop)
        ax.set_title(f"{event}\n情绪分布", fontproperties=font_prop)
    for j in range(len(events), 4):
        axes[j].axis('off')
    plt.suptitle("不同重大事件中情绪分布", fontsize=14, fontweight='bold', fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig("event_emotion_pie.png", dpi=300)
    plt.show()

def plot_event_wordclouds(df: pd.DataFrame) -> None:
    from wordcloud import WordCloud
    import re
    font_prop = get_font_properties()
    if "primary_event" not in df.columns:
        return
    text_col = None
    for col in ["text", "tweet_content", "content", "tweet_text"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        return
    events = [e for e in df["primary_event"].unique() if e != "无关联事件"][:4]
    if len(events) == 0:
        return
    stopwords = set([
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
        'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
        'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
        'our', 'their', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when',
        'why', 'how', 'all', 'each', 'every', 'any', 'some', 'no', 'none',
        'http', 'https', 'rt', 'amp', 'co', 'de', 'la', 'le', 'les', 'un', 'une',
        'et', 'est', 'en', 'du', 'des', 'au', 'aux', 'ce', 'cette', 'qui', 'que',
        'pour', 'sur', 'avec', 'dans', 'par', 'pas', 'plus', 'tout', 'tous',
        'ne', 'se', 'son', 'sa', 'ses', 'leur', 'leurs', 'nous', 'vous', 'ils',
    ])
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    for i, event in enumerate(events):
        ax = axes[i]
        event_data = df[df["primary_event"] == event][text_col]
        all_text = ' '.join(event_data.dropna().astype(str).tolist())
        all_text = re.sub(r'http\S+', '', all_text)
        all_text = re.sub(r'@\w+', '', all_text)
        all_text = re.sub(r'#\w+', '', all_text)
        all_text = re.sub(r'[^\w\s]', ' ', all_text)
        all_text = re.sub(r'\d+', '', all_text)
        if len(all_text.strip()) < 10:
            ax.text(0.5, 0.5, f"{event}\n文本数据不足", ha='center', va='center',
                   fontsize=12, fontproperties=font_prop)
            ax.axis('off')
            continue
        try:
            wc = WordCloud(
                width=800, height=600,
                background_color='white',
                max_words=100,
                stopwords=stopwords,
                font_path=CHINESE_FONT_PATH if CHINESE_FONT_PATH else None,
                colormap='viridis',
                min_font_size=10,
                max_font_size=100,
            )
            wc.generate(all_text)
            ax.imshow(wc, interpolation='bilinear')
            ax.set_title(f"{event}\n词云图", fontsize=12, fontproperties=font_prop)
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f"{event}\n生成失败", ha='center', va='center',
                   fontsize=10, fontproperties=font_prop)
            ax.axis('off')
    for j in range(len(events), 4):
        axes[j].axis('off')
    plt.suptitle("不同重大事件的词云图", fontsize=14, fontweight='bold', fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig("event_wordclouds.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_histogram(df: pd.DataFrame) -> None:
    font_prop = get_font_properties()
    if "favorites_count" not in df.columns:
        return
    correlations = {}
    target_cols = [
        ("user_followers_count", "粉丝数"),
        ("tweet_length", "推文长度"),
        ("emotion_index", "情绪指数"),
        ("max_event_relevance", "事件关联度"),
        ("user_statuses_count", "发帖总数"),
    ]
    for col, label in target_cols:
        if col in df.columns:
            corr = df["favorites_count"].corr(df[col])
            if not pd.isna(corr):
                correlations[label] = corr
    if not correlations:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(correlations.keys())
    values = list(correlations.values())
    colors = ['green' if v > 0 else 'red' for v in values]
    bars = ax.barh(labels, values, color=colors, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel("皮尔逊相关系数", fontproperties=font_prop)
    ax.set_ylabel("因素", fontproperties=font_prop)
    ax.set_title("各因素与收藏量的相关系数", fontproperties=font_prop)
    for i, (label, v) in enumerate(zip(ax.get_yticklabels(), values)):
        label.set_fontproperties(font_prop)
        ax.text(v + 0.01 if v > 0 else v - 0.08, i, f'{v:.4f}', va='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig("correlation_histogram.png", dpi=300)
    plt.show()

def classify_event_sentiment(event: Dict) -> str:
    negative_keywords = ["attentat", "attack", "terror", "mort", "victime", "dead", 
                         "victim", "explosion", "shooting", "恐怖", "袭击", "死亡"]
    positive_keywords = ["victoire", "win", "champion", "fete", "celebration", 
                         "concert", "festival", "胜利", "冠军"]
    all_keywords = (
        event.get("keywords_fr", []) + 
        event.get("keywords_en", []) + 
        event.get("keywords_cn", [])
    )
    neg_count = sum(1 for kw in all_keywords if kw.lower() in [nk.lower() for nk in negative_keywords])
    pos_count = sum(1 for kw in all_keywords if kw.lower() in [pk.lower() for pk in positive_keywords])
    if neg_count > pos_count:
        return "消极事件"
    elif pos_count > neg_count:
        return "积极事件"
    else:
        return "中性事件"

def plot_event_sentiment_prediction(df: pd.DataFrame) -> None:
    font_prop = get_font_properties()
    if "posted_datetime" not in df.columns:
        return
    if "emotion_index" not in df.columns or "max_event_relevance" not in df.columns:
        return
    event_sentiments = {}
    for event in MAJOR_EVENTS:
        sentiment = classify_event_sentiment(event)
        event_sentiments[event["name_cn"]] = sentiment
    df_copy = df.copy()
    df_copy["date"] = df_copy["posted_datetime"].dt.date
    daily_stats = df_copy.groupby("date").agg({
        "emotion_index": "mean",
        "max_event_relevance": "mean",
        "tweet_id": "count",
        "favorites_count": "mean"
    }).reset_index()
    daily_stats.columns = ["date", "avg_emotion", "avg_relevance", "tweet_count", "avg_favorites"]
    daily_stats["date"] = pd.to_datetime(daily_stats["date"])
    daily_stats = daily_stats.sort_values("date")
    start_date = pd.to_datetime("2015-11-01")
    end_date = pd.to_datetime("2016-01-15")
    daily_stats = daily_stats[(daily_stats["date"] >= start_date) & (daily_stats["date"] <= end_date)]
    if len(daily_stats) == 0:
        return
    relevance_threshold = daily_stats["avg_relevance"].mean() + 2 * daily_stats["avg_relevance"].std()
    tweet_count_threshold = daily_stats["tweet_count"].mean() + 2 * daily_stats["tweet_count"].std()
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(daily_stats["date"], daily_stats["avg_emotion"], 
                     color="blue", linewidth=2, marker="o", markersize=4, label="情绪指数")
    line2 = ax1_twin.plot(daily_stats["date"], daily_stats["avg_relevance"], 
                          color="red", linewidth=2, marker="s", markersize=4, label="事件关联度")
    ax1_twin.axhline(y=relevance_threshold, color="red", linestyle="--", linewidth=2, alpha=0.8, label=f"重大事件阈值({relevance_threshold:.4f})")
    ax1.set_xlabel("", fontproperties=font_prop)
    ax1.set_ylabel("情绪指数", color="blue", fontproperties=font_prop)
    ax1_twin.set_ylabel("事件关联度", color="red", fontproperties=font_prop)
    ax1.set_title("情绪指数与事件关联度随时间变化 (2015.11-2016.01)", fontproperties=font_prop)
    ax1.set_xticklabels([])
    lines = line1 + line2
    labels = ["情绪指数", "事件关联度"]
    ax1.legend(lines, labels, loc="upper left", prop=font_prop)
    ax1.grid(True, alpha=0.3)
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    ax2.bar(daily_stats["date"], daily_stats["tweet_count"], color="steelblue", alpha=0.6, label="发帖量")
    ax2_twin.plot(daily_stats["date"], daily_stats["avg_favorites"], 
                  color="orange", linewidth=2, marker="^", markersize=4, label="平均收藏量")
    ax2.axhline(y=tweet_count_threshold, color="red", linestyle="--", linewidth=2, alpha=0.8, label=f"流量阈值({tweet_count_threshold:.0f})")
    ax2.set_xlabel("", fontproperties=font_prop)
    ax2.set_ylabel("发帖量", color="steelblue", fontproperties=font_prop)
    ax2_twin.set_ylabel("平均收藏量", color="orange", fontproperties=font_prop)
    ax2.set_title("流量预测：发帖量与收藏量变化 (2015.11-2016.01)", fontproperties=font_prop)
    ax2.set_xticklabels([])
    ax2.legend(loc="upper left", prop=font_prop)
    ax2_twin.legend(loc="upper right", prop=font_prop)
    ax2.grid(True, alpha=0.3)
    ax3 = axes[1, 0]
    negative_events = [e for e in MAJOR_EVENTS if event_sentiments.get(e["name_cn"]) == "消极事件"]
    if negative_events:
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(negative_events)))
        for i, event in enumerate(negative_events):
            try:
                peak_date = pd.to_datetime(event["peak_date"])
                mask = (daily_stats["date"] >= peak_date - pd.Timedelta(days=7)) & \
                       (daily_stats["date"] <= peak_date + pd.Timedelta(days=7))
                event_data = daily_stats[mask].copy()
                if len(event_data) > 0:
                    event_data["relative_day"] = (event_data["date"] - peak_date).dt.days
                    ax3.plot(event_data["relative_day"], event_data["avg_emotion"], 
                            color=colors[i], linewidth=2, marker="o", markersize=5,
                            label=event["name_cn"][:10])
            except:
                pass
        ax3.axvline(x=0, color="black", linestyle="--", linewidth=2, alpha=0.8)
        ax3.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax3.set_xlabel("相对事件峰值的天数", fontproperties=font_prop)
        ax3.set_ylabel("平均情绪指数", fontproperties=font_prop)
        ax3.set_title("舆论预警：消极事件前后情绪变化", fontproperties=font_prop)
        ax3.legend(prop=font_prop, loc="best")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "无消极事件数据", ha="center", va="center", 
                fontsize=14, fontproperties=font_prop)
        ax3.set_title("舆论预警：消极事件前后情绪变化", fontproperties=font_prop)
    ax4 = axes[1, 1]
    if negative_events:
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(negative_events)))
        for i, event in enumerate(negative_events):
            try:
                peak_date = pd.to_datetime(event["peak_date"])
                event_name = event["name_cn"].replace(" ", "_")
                relevance_col = f"event_relevance_{event_name}"
                if relevance_col in df_copy.columns:
                    event_daily = df_copy.groupby("date")[relevance_col].mean().reset_index()
                    event_daily.columns = ["date", "relevance"]
                    event_daily["date"] = pd.to_datetime(event_daily["date"])
                    mask = (event_daily["date"] >= peak_date - pd.Timedelta(days=7)) & \
                           (event_daily["date"] <= peak_date + pd.Timedelta(days=7))
                    event_data = event_daily[mask].copy()
                    if len(event_data) > 0:
                        event_data["relative_day"] = (event_data["date"] - peak_date).dt.days
                        ax4.plot(event_data["relative_day"], event_data["relevance"], 
                                color=colors[i], linewidth=2, marker="s", markersize=5,
                                label=event["name_cn"][:10])
            except:
                pass
        ax4.axvline(x=0, color="black", linestyle="--", linewidth=2, alpha=0.8)
        ax4.set_xlabel("相对事件峰值的天数", fontproperties=font_prop)
        ax4.set_ylabel("事件关联度", fontproperties=font_prop)
        ax4.set_title("舆论预警：消极事件关联度变化曲线", fontproperties=font_prop)
        ax4.legend(prop=font_prop, loc="best")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "无消极事件数据", ha="center", va="center", 
                fontsize=14, fontproperties=font_prop)
        ax4.set_title("舆论预警：消极事件关联度变化曲线", fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig("event_sentiment_prediction.png", dpi=300, bbox_inches='tight')
    plt.show()

def run_all_analysis(csv_path: str | None = None) -> None:
    df = load_tweet_data(csv_path)
    df = add_derived_features(df)
    df = add_event_features(df)
    plot_event_histogram(df)
    analyze_event_relevance_distribution(df)
    plot_event_relevance_over_time(df)
    df = add_advanced_features(df)
    if "user_followers_count" in df.columns and "fav_status_ratio" in df.columns:
        plot_quadrant_scatter(
            df["user_followers_count"], df["fav_status_ratio"],
            "粉丝数", "收藏量/发帖量比",
            "粉丝数 vs 收藏效率", "quadrant_followers_fav_ratio.png"
        )
    if "max_event_relevance" in df.columns and "total_reply_activity" in df.columns:
        plot_quadrant_scatter(
            df["max_event_relevance"], df["total_reply_activity"],
            "事件关联度", "回复活跃度",
            "事件关联度 vs 回复活跃度", "quadrant_relevance_reply.png",
            center_origin=True
        )
    plot_radar_chart(df)
    plot_event_emotion_pie_charts(df)
    plot_event_wordclouds(df)
    plot_correlation_histogram(df)
    plot_reply_network_force_directed(df)
    if "primary_event" in df.columns:
        events = [e for e in df["primary_event"].unique() if e != "无关联事件"]
        for event in events[:2]:
            plot_reply_network_force_directed(df, event_name=event)
    plot_event_propagation_tree(df)
    plot_event_sentiment_prediction(df)

if __name__ == "__main__":
    run_all_analysis()
