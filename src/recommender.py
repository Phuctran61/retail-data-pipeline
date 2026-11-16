import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_product_recommendations(
    db_path: Path, min_buyers: int = 20, top_n: int = 5
) -> pd.DataFrame:
    """
    Xây dựng hệ thống gợi ý sản phẩm mua kèm (Item-based Collaborative Filtering)
    sử dụng độ tương đồng Cosine trên ma trận tương tác Khách hàng - Sản phẩm.
    """
    with sqlite3.connect(db_path) as conn:
        interactions = pd.read_sql_query(
            """
            SELECT customer_id, stock_code, COUNT(DISTINCT invoice_no) AS purchase_count
            FROM fact_sales
            GROUP BY customer_id, stock_code
            """,
            conn,
        )
        dim_products = pd.read_sql_query(
            "SELECT stock_code, description FROM dim_products", conn
        )

    # 1. Lọc nhiễu: Chỉ giữ các sản phẩm có ít nhất `min_buyers` khách hàng khác nhau từng mua
    item_buyer_counts = interactions.groupby("stock_code")["customer_id"].nunique()
    popular_items = item_buyer_counts[item_buyer_counts >= min_buyers].index
    filtered = interactions[interactions["stock_code"].isin(popular_items)]

    print(f"Tổng số sản phẩm trong kho        : {len(item_buyer_counts):,}")
    print(f"Sản phẩm đủ ngưỡng (>= {min_buyers} KH mua) : {len(popular_items):,}")

    # 2. Tạo ma trận nhị phân Customer x Product (1 = đã từng mua, 0 = chưa mua)
    user_item_matrix = filtered.pivot_table(
        index="customer_id",
        columns="stock_code",
        values="purchase_count",
        fill_value=0,
    )
    user_item_binary = (user_item_matrix > 0).astype(np.float32)

    # 3. Tính ma trận Cosine Similarity giữa các cột (sản phẩm với sản phẩm)
    sim_matrix = cosine_similarity(user_item_binary.T)
    sim_df = pd.DataFrame(
        sim_matrix,
        index=user_item_binary.columns,
        columns=user_item_binary.columns,
    )

    # 4. Trích xuất Top N sản phẩm gợi ý cho từng mã hàng
    prod_lookup = dim_products.set_index("stock_code")["description"].to_dict()
    rec_records = []

    for stock_code in sim_df.index:
        top_similar = sim_df[stock_code].drop(stock_code).nlargest(top_n)
        for rank, (rec_code, score) in enumerate(top_similar.items(), start=1):
            rec_records.append(
                {
                    "stock_code": stock_code,
                    "product_name": prod_lookup.get(stock_code, "UNKNOWN"),
                    "rank": rank,
                    "recommended_stock_code": rec_code,
                    "recommended_product_name": prod_lookup.get(rec_code, "UNKNOWN"),
                    "similarity_score": round(float(score), 4),
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

    # Kiểm thử thực tế trên 2 sản phẩm bán chạy nhất
    for sample_code in ["22423", "85123A"]:
        sample = recommendations[recommendations["stock_code"] == sample_code]
        prod_title = sample["product_name"].iloc[0]
        print(f"\n=== TOP 5 GỢI Ý MUA KÈM CHO: [{sample_code}] {prod_title} ===")
        print(
            sample[
                ["rank", "recommended_stock_code", "recommended_product_name", "similarity_score"]
            ].to_string(index=False)
        )