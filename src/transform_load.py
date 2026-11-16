import sqlite3
from pathlib import Path
import pandas as pd


def extract_raw(filepath: Path) -> pd.DataFrame:
    """Đọc dữ liệu thô với encoding ISO-8859-1 và schema định sẵn."""
    dtype_schema = {
        "Invoice": "string",
        "StockCode": "string",
        "Description": "string",
        "Quantity": "int64",
        "InvoiceDate": "string",
        "Price": "float64",
        "Customer ID": "float64",
        "Country": "string",
    }
    return pd.read_csv(filepath, encoding="ISO-8859-1", dtype=dtype_schema)


def transform_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dữ liệu thô qua 5 bước lọc có kiểm toán (Audit Logging)
    và chuẩn hóa kiểu dữ liệu.
    """
    df = df_raw.copy()
    print(f"[1/5] Dữ liệu thô ban đầu              : {len(df):,} dòng")

    # Chuẩn hóa khoảng trắng đầu/cuối chuỗi
    for col in ["Invoice", "StockCode", "Description", "Country"]:
        df[col] = df[col].str.strip()

    # Bước 1: Loại bỏ bản ghi trùng lặp 100%
    df = df.drop_duplicates()
    print(f"[2/5] Sau khi khử trùng lặp           : {len(df):,} dòng")

    # Bước 2: Loại bỏ dòng thiếu Customer ID (bắt buộc cho RFM & Recommendation)
    df = df.dropna(subset=["Customer ID"])
    print(f"[3/5] Sau khi lọc thiếu Customer ID   : {len(df):,} dòng")

    # Bước 3: Loại bỏ hóa đơn hủy ('C') và giao dịch có số lượng / đơn giá <= 0
    df = df[~df["Invoice"].str.startswith("C", na=False)]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    print(f"[4/5] Sau khi lọc đơn hủy & số âm     : {len(df):,} dòng")

    # Bước 4: Chỉ giữ mã sản phẩm thực tế (bắt đầu bằng 5 chữ số)
    df = df[df["StockCode"].str.match(r"^\d{5}", na=False)]
    print(f"[5/5] Sau khi lọc mã phi sản phẩm     : {len(df):,} dòng")

    # Bước 5: Ép kiểu chuẩn và tạo cột doanh thu
    df["Customer ID"] = df["Customer ID"].astype("int64")
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], format="%m/%d/%Y %H:%M"
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["Description"] = df["Description"].fillna("UNKNOWN").str.upper()
    df["TotalAmount"] = (df["Quantity"] * df["Price"]).round(2)

    return df


def build_star_schema(
    df_clean: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tách dữ liệu sạch thành 2 bảng Dimension và 1 bảng Fact chuẩn khóa chính."""
    # Bảng chiều Khách hàng (1 dòng duy nhất cho mỗi customer_id)
    dim_customers = (
        df_clean.groupby("Customer ID", as_index=False)
        .agg(country=("Country", lambda s: s.mode().iloc[0]))
        .rename(columns={"Customer ID": "customer_id"})
    )

    # Bảng chiều Sản phẩm (1 dòng duy nhất cho mỗi stock_code)
    dim_products = (
        df_clean.groupby("StockCode", as_index=False)
        .agg(
            description=("Description", lambda s: s.mode().iloc[0]),
            avg_unit_price=("Price", lambda x: round(x.mean(), 2)),
        )
        .rename(columns={"StockCode": "stock_code"})
    )

    # Bảng sự kiện Bán hàng
    fact_sales = (
        df_clean[
            [
                "Invoice",
                "StockCode",
                "Customer ID",
                "InvoiceDate",
                "Quantity",
                "Price",
                "TotalAmount",
            ]
        ]
        .rename(
            columns={
                "Invoice": "invoice_no",
                "StockCode": "stock_code",
                "Customer ID": "customer_id",
                "InvoiceDate": "invoice_date",
                "Quantity": "quantity",
                "Price": "unit_price",
                "TotalAmount": "total_amount",
            }
        )
        .reset_index(drop=True)
    )
    fact_sales.insert(0, "sale_id", fact_sales.index + 1)

    # Data Quality Assertions trước khi Load vào Database
    assert dim_customers["customer_id"].is_unique, "Lỗi: Trùng khóa chính dim_customers!"
    assert dim_products["stock_code"].is_unique, "Lỗi: Trùng khóa chính dim_products!"
    assert (fact_sales["total_amount"] <= 0).sum() == 0, "Lỗi: Tồn tại doanh thu <= 0!"

    return dim_customers, dim_products, fact_sales


def load_to_sqlite(
    dim_customers: pd.DataFrame,
    dim_products: pd.DataFrame,
    fact_sales: pd.DataFrame,
    db_path: Path,
) -> None:
    """Nạp các bảng Star Schema vào cơ sở dữ liệu SQLite và tạo Index."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        dim_customers.to_sql("dim_customers", conn, if_exists="replace", index=False)
        dim_products.to_sql("dim_products", conn, if_exists="replace", index=False)
        fact_sales.to_sql("fact_sales", conn, if_exists="replace", index=False)

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fact_customer ON fact_sales(customer_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fact_stock ON fact_sales(stock_code);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fact_date ON fact_sales(invoice_date);"
        )

    print(f"\n[LOAD THÀNH CÔNG] Đã lưu Data Warehouse tại: {db_path}")
    print(f" - dim_customers : {len(dim_customers):,} khách hàng")
    print(f" - dim_products  : {len(dim_products):,} sản phẩm")
    print(
        f" - fact_sales    : {len(fact_sales):,} dòng giao dịch (£{fact_sales['total_amount'].sum():,.2f})"
    )


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RAW_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail_II.csv"
    DB_PATH = PROJECT_ROOT / "data" / "processed" / "retail_dw.db"

    raw_df = extract_raw(RAW_PATH)
    clean_df = transform_data(raw_df)
    dim_cust, dim_prod, fact = build_star_schema(clean_df)
    load_to_sqlite(dim_cust, dim_prod, fact, DB_PATH)