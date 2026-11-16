from pathlib import Path
from src.ingestion_profiling import load_raw_data, run_data_profiling
from src.transform_load import transform_data, build_star_schema, load_to_sqlite
from src.build_rfm_mart import execute_rfm_sql
from src.clustering_rfm import load_rfm_mart, preprocess_features, train_and_save_clusters
from src.recommender import build_product_recommendations
from src.generate_charts import plot_elbow_curve, plot_cluster_revenue, plot_top_products


def run_full_pipeline() -> None:
    project_root = Path(__file__).resolve().parent
    raw_path = project_root / "data" / "raw" / "online_retail_II.csv"
    db_path = project_root / "data" / "processed" / "retail_dw.db"
    sql_path = project_root / "sql" / "rfm_analysis.sql"
    assets_dir = project_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    print("BƯỚC 1: KIỂM ĐỊNH DỮ LIỆU THÔ (DATA PROFILING)")
    raw_df = load_raw_data(raw_path)
    audit = run_data_profiling(raw_df)
    print(f"-> Đã kiểm định {audit['total_rows']:,} dòng dữ liệu thô.\n")

    print("BƯỚC 2: LÀM SẠCH & NẠP STAR SCHEMA VÀO DATA WAREHOUSE")
    clean_df = transform_data(raw_df)
    dim_cust, dim_prod, fact = build_star_schema(clean_df)
    load_to_sqlite(dim_cust, dim_prod, fact, db_path)
    print()

    print("BƯỚC 3: TẠO BẢNG TỔNG HỢP RFM BẰNG SQL")
    execute_rfm_sql(db_path, sql_path)
    print()

    print("BƯỚC 4: PHÂN CỤM KHÁCH HÀNG K-MEANS")
    rfm_df = load_rfm_mart(db_path)
    X_scaled = preprocess_features(rfm_df)
    train_and_save_clusters(rfm_df, X_scaled, db_path)
    print()

    print("BƯỚC 5: XÂY DỰNG HỆ THỐNG GỢI Ý SẢN PHẨM")
    build_product_recommendations(db_path, min_buyers=20, top_n=5)
    print()

    print("BƯỚC 6: XUẤT BIỂU ĐỒ BÁO CÁO TRỰC QUAN")
    plot_elbow_curve(db_path, assets_dir)
    plot_cluster_revenue(db_path, assets_dir)
    plot_top_products(db_path, assets_dir)
    print("\n=== HOÀN TẤT TOÀN BỘ DATA PIPELINE ===")


if __name__ == "__main__":
    run_full_pipeline()