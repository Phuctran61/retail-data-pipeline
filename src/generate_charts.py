import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cấu hình giao diện biểu đồ sạch, sắc nét chuẩn báo cáo
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


def plot_elbow_curve(db_path: Path, assets_dir: Path) -> None:
    """Biểu đồ 1: Đường cong Elbow chọn số cụm k tối ưu cho K-Means."""
    with sqlite3.connect(db_path) as conn:
        rfm_df = pd.read_sql_query(
            "SELECT recency, frequency, monetary FROM mart_customer_rfm", conn
        )

    X_scaled = StandardScaler().fit_transform(np.log1p(rfm_df))
    k_values = list(range(2, 8))
    inertias = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k_values, inertias, marker="o", linewidth=2.5, markersize=8, color="#2563EB")
    ax.axvline(x=4, color="#DC2626", linestyle="--", linewidth=1.5, label="Optimal k = 4 (Elbow Point)")
    ax.scatter([4], [inertias[2]], color="#DC2626", s=120, zorder=5)

    ax.set_title("Elbow Method for Optimal K-Means Clusters", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=11)
    ax.legend(frameon=True)

    fig.savefig(assets_dir / "1_kmeans_elbow.png")
    plt.close(fig)


def plot_cluster_revenue(db_path: Path, assets_dir: Path) -> None:
    """Biểu đồ 2: So sánh tỷ trọng Khách hàng vs Tỷ trọng Doanh thu theo cụm K-Means."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT cluster_label, monetary FROM mart_customer_clusters", conn)

    summary = (
        df.groupby("cluster_label", as_index=False)
        .agg(customers=("monetary", "count"), revenue=("monetary", "sum"))
        .sort_values(by="revenue", ascending=False)
    )
    summary["cust_pct"] = summary["customers"] / summary["customers"].sum() * 100
    summary["rev_pct"] = summary["revenue"] / summary["revenue"].sum() * 100

    x = np.arange(len(summary))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, summary["cust_pct"], width, label="% Customers", color="#64748B")
    bars2 = ax.bar(x + width / 2, summary["rev_pct"], width, label="% Total Revenue", color="#10B981")

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9.5)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold")

    ax.set_title("Customer Share vs. Revenue Contribution by K-Means Segment", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["cluster_label"], fontsize=10.5)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_ylim(0, max(summary["rev_pct"]) + 12)
    ax.legend(frameon=True)

    fig.savefig(assets_dir / "2_cluster_revenue_share.png")
    plt.close(fig)


def plot_top_products(db_path: Path, assets_dir: Path) -> None:
    """Biểu đồ 3: Top 10 sản phẩm mang lại doanh thu cao nhất từ Star Schema."""
    with sqlite3.connect(db_path) as conn:
        top_df = pd.read_sql_query(
            """
            SELECT 
                p.stock_code || ' - ' || SUBSTR(p.description, 1, 28) AS product_label,
                ROUND(SUM(f.total_amount) / 1000.0, 2) AS revenue_k
            FROM fact_sales f
            JOIN dim_products p ON f.stock_code = p.stock_code
            GROUP BY p.stock_code, p.description
            ORDER BY revenue_k DESC
            LIMIT 10;
            """,
            conn,
        )

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    sns.barplot(data=top_df, x="revenue_k", y="product_label", palette="Blues_r", hue="product_label", legend=False, ax=ax)

    for i, v in enumerate(top_df["revenue_k"]):
        ax.text(v + 2, i, f"£{v:,.1f}K", va="center", fontsize=9.5, fontweight="bold")

    ax.set_title("Top 10 Revenue-Generating Products (£ Thousands)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Total Revenue (£K)", fontsize=11)
    ax.set_ylabel("")
    ax.set_xlim(0, top_df["revenue_k"].max() * 1.15)

    fig.savefig(assets_dir / "3_top_10_products.png")
    plt.close(fig)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DB_PATH = PROJECT_ROOT / "data" / "processed" / "retail_dw.db"
    ASSETS_DIR = PROJECT_ROOT / "assets"
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    plot_elbow_curve(DB_PATH, ASSETS_DIR)
    plot_cluster_revenue(DB_PATH, ASSETS_DIR)
    plot_top_products(DB_PATH, ASSETS_DIR)
    print(f"[THÀNH CÔNG] Đã xuất 3 biểu đồ chất lượng cao vào thư mục: {ASSETS_DIR}")