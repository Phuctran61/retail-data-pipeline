import sqlite3
from pathlib import Path
import pandas as pd


def execute_rfm_sql(db_path: Path, sql_path: Path) -> None:
    """Đọc tệp SQL thuần, tạo bảng mart_customer_rfm và kiểm tra kết quả."""
    sql_script = sql_path.read_text(encoding="utf-8")
    if not sql_script.strip():
        raise ValueError(f"Tệp SQL đang bị trống hoặc chưa được lưu: {sql_path}")

    with sqlite3.connect(db_path) as conn:
        conn.executescript(sql_script)

        # Truy vấn kiểm tra phân bố các nhóm khách hàng
        summary_df = pd.read_sql_query(
            """
            SELECT 
                rfm_segment AS Phan_khuc,
                COUNT(*) AS So_khach_hang,
                ROUND(AVG(recency), 1) AS Recency_TB,
                ROUND(AVG(frequency), 1) AS Frequency_TB,
                ROUND(AVG(monetary), 2) AS Monetary_TB,
                ROUND(SUM(monetary), 2) AS Tong_doanh_thu
            FROM mart_customer_rfm
            GROUP BY rfm_segment
            ORDER BY Tong_doanh_thu DESC;
            """,
            conn,
        )

        # Truy vấn Top 5 sản phẩm mang lại doanh thu cao nhất (Product Analytics)
        top_products_df = pd.read_sql_query(
            """
            SELECT 
                p.stock_code,
                p.description,
                SUM(f.quantity) AS Tong_so_luong,
                ROUND(SUM(f.total_amount), 2) AS Tong_doanh_thu
            FROM fact_sales f
            JOIN dim_products p ON f.stock_code = p.stock_code
            GROUP BY p.stock_code, p.description
            ORDER BY Tong_doanh_thu DESC
            LIMIT 5;
            """,
            conn,
        )

    print("=== TỔNG HỢP PHÂN KHÚC RFM (TỪ SQL DATA MART) ===")
    print(summary_df.to_string(index=False))
    print("\n=== TOP 5 SẢN PHẨM DOANH THU CAO NHẤT ===")
    print(top_products_df.to_string(index=False))


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DB_PATH = PROJECT_ROOT / "data" / "processed" / "retail_dw.db"
    SQL_PATH = PROJECT_ROOT / "sql" / "rfm_analysis.sql"

    execute_rfm_sql(DB_PATH, SQL_PATH)