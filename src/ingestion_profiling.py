from pathlib import Path
import pandas as pd


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """
    Đọc dữ liệu thô với bảng mã ISO-8859-1 và ép kiểu dữ liệu (Schema) rõ ràng
    để tránh lỗi mixed-type khi Pandas tự suy đoán.
    """
    dtype_schema = {
        "Invoice": "string",
        "StockCode": "string",
        "Description": "string",
        "Quantity": "int64",
        "InvoiceDate": "string",
        "Price": "float64",
        "Customer ID": "float64",  # Dùng float64 tạm thời vì cột chứa NaN
        "Country": "string",
    }
    return pd.read_csv(filepath, encoding="ISO-8859-1", dtype=dtype_schema)


def run_data_profiling(df: pd.DataFrame) -> dict:
    """
    Kiểm định chất lượng dữ liệu thô trên 4 khía cạnh:
    Tính đầy đủ, Tính duy nhất, Quy tắc nghiệp vụ và Mã phi sản phẩm.
    """
    total_rows = len(df)

    # 1. Completeness: Kiểm tra giá trị khuyết
    missing_cust = df["Customer ID"].isna().sum()
    missing_desc = df["Description"].isna().sum()

    # 2. Uniqueness: Kiểm tra trùng lặp hoàn toàn
    exact_dups = df.duplicated().sum()

    # 3. Validity: Kiểm tra quy tắc giao dịch hợp lệ
    cancelled_inv = df["Invoice"].str.strip().str.startswith("C", na=False).sum()
    invalid_qty = (df["Quantity"] <= 0).sum()
    invalid_price = (df["Price"] <= 0).sum()

    # 4. Kiểm tra mã phi sản phẩm (StockCode chuẩn luôn bắt đầu bằng 5 chữ số)
    non_product_mask = ~df["StockCode"].str.strip().str.match(r"^\d{5}", na=False)
    non_product_rows = non_product_mask.sum()
    top_non_products = (
        df.loc[non_product_mask, "StockCode"].value_counts().head(8).to_dict()
    )

    return {
        "total_rows": total_rows,
        "missing_cust": int(missing_cust),
        "missing_desc": int(missing_desc),
        "exact_dups": int(exact_dups),
        "cancelled_inv": int(cancelled_inv),
        "invalid_qty": int(invalid_qty),
        "invalid_price": int(invalid_price),
        "non_product_rows": int(non_product_rows),
        "top_non_products": top_non_products,
    }


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    FILE_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail_II.csv"

    df_raw = load_raw_data(FILE_PATH)
    audit = run_data_profiling(df_raw)

    print("=== BÁO CÁO KIỂM ĐỊNH DỮ LIỆU THÔ (DATA PROFILING) ===")
    print(f"Tổng số bản ghi thô          : {audit['total_rows']:,}")
    print(f"1. Thiếu Customer ID         : {audit['missing_cust']:,} ({audit['missing_cust']/audit['total_rows']*100:.2f}%)")
    print(f"2. Thiếu Description         : {audit['missing_desc']:,} ({audit['missing_desc']/audit['total_rows']*100:.2f}%)")
    print(f"3. Bản ghi trùng lặp 100%    : {audit['exact_dups']:,} ({audit['exact_dups']/audit['total_rows']*100:.2f}%)")
    print(f"4. Hóa đơn hủy (bắt đầu 'C') : {audit['cancelled_inv']:,} ({audit['cancelled_inv']/audit['total_rows']*100:.2f}%)")
    print(f"5. Số lượng <= 0             : {audit['invalid_qty']:,} ({audit['invalid_qty']/audit['total_rows']*100:.2f}%)")
    print(f"6. Đơn giá <= 0              : {audit['invalid_price']:,} ({audit['invalid_price']/audit['total_rows']*100:.2f}%)")
    print(f"7. Mã phi sản phẩm           : {audit['non_product_rows']:,} ({audit['non_product_rows']/audit['total_rows']*100:.2f}%)")
    print(f"-> Chi tiết mã phi SP     : {audit['top_non_products']}")