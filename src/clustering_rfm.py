import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def load_rfm_mart(db_path: Path) -> pd.DataFrame:
    """Đọc bảng mart_customer_rfm từ SQLite Data Warehouse."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM mart_customer_rfm", conn)


def preprocess_features(df: pd.DataFrame) -> np.ndarray:
    """
    Xử lý phân phối lệch phải bằng Log(x + 1) và chuẩn hóa Z-score (StandardScaler).
    """
    features = df[["recency", "frequency", "monetary"]]
    log_transformed = np.log1p(features)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(log_transformed)
    return scaled_features


def evaluate_kmeans(scaled_data: np.ndarray, k_range: range) -> None:
    """Đánh giá mô hình qua Inertia (Elbow) và Silhouette Score."""
    print("=== ĐÁNH GIÁ SỐ CỤM K TỐI ƯU ===")
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled_data)
        sil = silhouette_score(scaled_data, labels)
        print(f"k = {k} | Inertia = {model.inertia_:8,.2f} | Silhouette Score = {sil:.4f}")


def train_and_save_clusters(df: pd.DataFrame, scaled_data: np.ndarray, db_path: Path) -> None:
    """Huấn luyện K-Means (k=4), gán nhãn động theo đặc trưng cụm và nạp vào Database."""
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster_id"] = kmeans.fit_predict(scaled_data)

    # --- PHẦN ĐÃ SỬA: Gán nhãn động theo đặc trưng trung bình của từng cụm ---
    stats = df.groupby("cluster_id")[["recency", "monetary"]].mean()
    vip_id = stats["monetary"].idxmax()   # Chi tiêu cao nhất -> VIP / Champions
    lost_id = stats["recency"].idxmax()   # Lâu không quay lại nhất -> Hibernating / Lost

    remaining = stats.drop([vip_id, lost_id])
    loyal_id = remaining["monetary"].idxmax()  # Chi tiêu cao nhì -> Loyal / Regular
    new_id = remaining["monetary"].idxmin()    # Mới mua, chi tiêu thấp -> New / Promising

    cluster_names = {
        vip_id: "VIP / Champions",
        loyal_id: "Loyal / Regular",
        new_id: "New / Promising",
        lost_id: "Hibernating / Lost",
    }
    df["cluster_label"] = df["cluster_id"].map(cluster_names)
    # -----------------------------------------------------------------------

    summary = (
        df.groupby(["cluster_id", "cluster_label"], as_index=False)
        .agg(
            So_khach_hang=("customer_id", "count"),
            Recency_TB=("recency", "mean"),
            Frequency_TB=("frequency", "mean"),
            Monetary_TB=("monetary", "mean"),
            Tong_doanh_thu=("monetary", "sum"),
        )
        .round(2)
        .sort_values(by="Tong_doanh_thu", ascending=False)
    )

    print("\n=== KẾT QUẢ PHÂN CỤM K-MEANS (K = 4) ===")
    print(summary.to_string(index=False))

    with sqlite3.connect(db_path) as conn:
        df.to_sql("mart_customer_clusters", conn, if_exists="replace", index=False)
    print(f"\n[LƯU THÀNH CÔNG] Đã ghi bảng 'mart_customer_clusters' ({len(df):,} khách hàng) vào {db_path.name}")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DB_PATH = PROJECT_ROOT / "data" / "processed" / "retail_dw.db"

    rfm_df = load_rfm_mart(DB_PATH)
    X_scaled = preprocess_features(rfm_df)
    evaluate_kmeans(X_scaled, range(2, 7))
    train_and_save_clusters(rfm_df, X_scaled, DB_PATH)