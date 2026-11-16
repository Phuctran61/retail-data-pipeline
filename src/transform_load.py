from collections import deque
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


def _remove_matched_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ghép cặp 1-1 theo thứ tự thời gian (FIFO) giữa giao dịch mua (+Q) và đơn hủy (-Q)
    có cùng (Customer ID, StockCode, Price, |Quantity|) với cancel_date >= purchase_date.
    """
    is_cancel = df["Invoice"].str.startswith("C", na=False) | (df["Quantity"] < 0)

    purchases = (
        df[~is_cancel & (df["Quantity"] > 0) & (df["Price"] > 0)]
        .sort_values(["_invoice_dt", "Invoice"])
        .reset_index(drop=True)
    )
    cancels = (
        df[is_cancel & (df["Quantity"] < 0) & (df["Price"] > 0)]
        .sort_values(["_invoice_dt", "Invoice"])
        .reset_index(drop=True)
    )

    purchase_queues: dict[tuple[int, str, float, int], deque[tuple[int, pd.Timestamp]]] = {}
    for idx, row in purchases.iterrows():
        key = (
            int(row["Customer ID"]),
            str(row["StockCode"]),
            round(float(row["Price"]), 4),
            int(row["Quantity"]),
        )
        purchase_queues.setdefault(key, deque()).append((idx, row["_invoice_dt"]))

    matched_indices: list[int] = []
    for _, crow in cancels.iterrows():
        key = (
            int(crow["Customer ID"]),
            str(crow["StockCode"]),
            round(float(crow["Price"]), 4),
            int(abs(crow["Quantity"])),
        )
        queue = purchase_queues.get(key)
        if queue and crow["_invoice_dt"] >= queue[0][1]:
            p_idx, _ = queue.popleft()
            matched_indices.append(p_idx)

    print(f"    -> Đã khử {len(matched_indices):,} cặp giao dịch mua - hủy đối ứng")
    return purchases.drop(index=matched_indices).reset_index(drop=True)


def transform_data(df_raw: pd.DataFrame, max_price_ratio: float = 20.0) -> pd.DataFrame:
    """
    Làm sạch dữ liệu thô qua 6 bước kiểm toán (Audit Logging):
    khử trùng lặp, lọc thiếu ID, chuẩn hóa mã SKU, khử cặp mua-hủy và lọc ngoại lai đơn giá.
    """
    df = df_raw.copy()
    print(f"[1/6] Dữ liệu thô ban đầu                 : {len(df):,} dòng")

    for col in ["Invoice", "StockCode", "Description", "Country"]:
        df[col] = df[col].str.strip()

    # Bước 1: Loại bỏ bản ghi trùng lặp 100%
    df = df.drop_duplicates()
    print(f"[2/6] Sau khi khử trùng lặp              : {len(df):,} dòng")

    # Bước 2: Loại bỏ dòng thiếu Customer ID
    df = df.dropna(subset=["Customer ID"]).copy()
    df["Customer ID"] = df["Customer ID"].astype("int64")
    print(f"[3/6] Sau khi lọc thiếu Customer ID      : {len(df):,} dòng")

    # Bước 3: Chỉ giữ mã sản phẩm thực tế (5 chữ số đầu) và phân tích mốc thời gian
    df = df[df["StockCode"].str.match(r"^\d{5}", na=False)].copy()
    df["_invoice_dt"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
    print(f"[4/6] Sau khi lọc mã phi sản phẩm        : {len(df):,} dòng")

    # Bước 4: Khử cặp mua - hủy đối ứng trước khi loại bỏ hóa đơn 'C'
    df = _remove_matched_cancellations(df)
    print(f"[5/6] Sau khi khử cặp mua - hủy đối ứng  : {len(df):,} dòng")

    # Bước 5: Kiểm soát ngoại lai đơn giá theo từng SKU (Price / median_price)
    sku_median_price = df.groupby("StockCode")["Price"].transform("median")
    outlier_mask = (df["Price"] / sku_median_price) > max_price_ratio
    print(
        f"    -> Đã loại {int(outlier_mask.sum()):,} dòng ngoại lai đơn giá "
        f"(> {max_price_ratio:.0f}x trung vị SKU)"
    )
    df = df[~outlier_mask].copy()
    print(f"[6/6] Sau khi lọc ngoại lai đơn giá      : {len(df):,} dòng")

    # Bước 6: Chuẩn hóa định dạng đầu ra
    df["InvoiceDate"] = df["_invoice_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["Description"] = df["Description"].fillna("UNKNOWN").str.upper()
    df["TotalAmount"] = (df["Quantity"] * df["Price"]).round(2)
    df = df.drop(columns=["_invoice_dt"])

    return df


def build_star_schema(
    df_clean: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tách dữ liệu sạch thành 2 bảng Dimension và 1 bảng Fact chuẩn khóa chính."""
    dim_customers = (
        df_clean.groupby("Customer ID", as_index=False)
        .agg(country=("Country", lambda s: s.mode().iloc[0]))
        .rename(columns={"Customer ID": "customer_id"})
    )

    dim_products = (
        df_clean.groupby("StockCode", as_index=False)
        .agg(
            description=("Description", lambda s: s.mode().iloc[0]),
            avg_unit_price=("Price", lambda x: round(x.mean(), 2)),
        )
        .rename(columns={"StockCode": "stock_code"})
    )

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