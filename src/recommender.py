import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def build_product_recommendations(
    db_path: Path, min_buyers: int = 20, top_n: int = 5
) -> pd.DataFrame:
    """
    Xây dựng hệ thống gợi ý sản phẩm mua kèm (Item-based Collaborative Filtering)
    sử dụng ma trận thưa CSR (Compressed Sparse Row) và độ tương đồng Cosine.
    """
    with sqlite3.connect(db_path) as conn:
        interactions = pd.read_sql_query(
            """
            SELECT DISTINCT customer_id, stock_code
            FROM fact_sales
            """,
            conn,
        )
        dim_products = pd.read_sql_query(
            "SELECT stock_code, description FROM dim_products", conn
        )

    # 1. Lọc nhiễu: Chỉ giữ các sản phẩm có ít nhất `min_buyers` khách hàng khác nhau từng mua
    item_buyer_counts = interactions.groupby("stock_code")["customer_id"].nunique()
    popular_items = item_buyer_counts[item_buyer_counts >= min_buyers].index
    filtered = interactions[interactions["stock_code"].isin(popular_items)].copy()

    print(f"Tổng số sản phẩm trong kho        : {len(item_buyer_counts):,}")
    print(f"Sản phẩm đủ ngưỡng (>= {min_buyers} KH mua) : {len(popular_items):,}")

    # 2. Xây dựng ma trận thưa nhị phân Customer x Product bằng scipy.sparse.csr_matrix
    cust_cat = pd.Categorical(filtered["customer_id"])
    item_cat = pd.Categorical(filtered["stock_code"])

    user_item_sparse = csr_matrix(
        (
            np.ones(len(filtered), dtype=np.float32),
            (cust_cat.codes, item_cat.codes),
        ),
        shape=(len(cust_cat.categories), len(item_cat.categories)),
    )

    sparse_kb = (
        user_item_sparse.data.nbytes
        + user_item_sparse.indices.nbytes
        + user_item_sparse.indptr.nbytes
    ) / 1024
    print(
        f"Kích thước ma trận CSR            : {user_item_sparse.shape[0]:,} x {user_item_sparse.shape[1]:,} "
        f"({sparse_kb:,.1f} KB RAM)"
    )

    # 3. Tính ma trận Cosine Similarity trực tiếp trên ma trận thưa chuyển vị (Item x Customer)
    sim_matrix = cosine_similarity(user_item_sparse.T, dense_output=True)
    item_codes = item_cat.categories.to_numpy()

    # 4. Trích xuất Top N sản phẩm gợi ý cho từng mã hàng bằng argpartition (O(N) thay vì sort toàn bộ)
    prod_lookup = dim_products.set_index("stock_code")["description"].to_dict()
    rec_records = []

    for idx, stock_code in enumerate(item_codes):
        scores = sim_matrix[idx].copy()
        scores[idx] = -1.0  # Loại bỏ chính sản phẩm đang xét

        top_indices = np.argpartition(scores, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        for rank, rec_idx in enumerate(top_indices, start=1):
            rec_code = item_codes[rec_idx]
            rec_records.append(
                {
                    "stock_code": stock_code,
                    "product_name": prod_lookup.get(stock_code, "UNKNOWN"),
                    "rank": rank,
                    "recommended_stock_code": rec_code,
                    "recommended_product_name": prod_lookup.get(rec_code, "UNKNOWN"),
                    "similarity_score": round(float(scores[rec_idx]), 4),
                }
            )

    rec_df = pd.DataFrame(rec_records)

    # 5. Lưu kết quả vào Data Mart trong SQLite
    with sqlite3.connect(db_path) as conn:
        rec_df.to_sql("mart_product_recommendations", conn, if_exists="replace", index=False)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rec_stock ON mart_product_recommendations(stock_code);"
        )

    print(
        f"[LƯU THÀNH CÔNG] Đã ghi {len(rec_df):,} dòng gợi ý vào bảng 'mart_product_recommendations'."
    )
    return rec_df


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DB_PATH = PROJECT_ROOT / "data" / "processed" / "retail_dw.db"

    recommendations = build_product_recommendations(DB_PATH, min_buyers=20, top_n=5)

    for sample_code in ["22423", "85123A"]:
        sample = recommendations[recommendations["stock_code"] == sample_code]
        prod_title = sample["product_name"].iloc[0]
        print(f"\n=== TOP 5 GỢI Ý MUA KÈM CHO: [{sample_code}] {prod_title} ===")
        print(
            sample[
                ["rank", "recommended_stock_code", "recommended_product_name", "similarity_score"]
            ].to_string(index=False)
        )