# End-to-End Retail Data Engineering & Analytics Pipeline

Xây dựng pipeline xử lý dữ liệu bán lẻ trực tuyến (541.910 bản ghi giao dịch) từ dữ liệu thô đến mô hình kho dữ liệu **Star Schema**, phân tích **RFM**, phân cụm khách hàng **K-Means** và hệ thống gợi ý sản phẩm mua kèm (**Item-based Collaborative Filtering**).

## 1. Kiến trúc luồng dữ liệu (Data Architecture)

```text
[online_retail_II.csv] (541,910 rows, ISO-8859-1)
         │
         ▼  (1. Ingestion & Data Quality Profiling)
[Data Cleaning & Validation] -> Loại bỏ trùng lặp, đơn hủy ('C'), khuyết ID, mã phi SP
         │
         ▼  (2. Star Schema Modeling - 391,150 clean rows)
┌────────────────────────────────────────────────────────┐
│              SQLite Data Warehouse (retail_dw.db)      │
│  • Core Tables: dim_customers, dim_products, fact_sales│
│  • Data Marts : mart_customer_rfm (SQL CTE & NTILE)    │
│                 mart_customer_clusters (K-Means k=4)   │
│                 mart_product_recommendations (Cosine)  │
└────────────────────────────────────────────────────────┘
```

## 2. Kết quả chính đạt được
* **Data Quality & Warehousing:** Làm sạch 541.910 bản ghi thô, giữ lại **391.150 giao dịch hợp lệ** (£8.74M doanh thu) cho **4.334 khách hàng** và **3.659 sản phẩm**.
* **Customer Segmentation (K-Means):** Chuẩn hóa đặc trưng RFM (`Log1p` + `StandardScaler`), xác định 4 cụm hành vi khách hàng; nhóm **VIP / Champions** (687 khách hàng — 15.8%) đóng góp **64.0% tổng doanh thu** (£5.59M).
* **Product Recommendation:** Tính toán ma trận tương đồng Cosine trên **2.399 sản phẩm phổ biến**, lưu trữ trước **11.995 gợi ý mua kèm** phục vụ truy vấn nhanh.

## 3. Hướng dẫn chạy dự án
```bash
pip install -r requirements.txt
# Đặt tệp online_retail_II.csv vào thư mục data/raw/
python main.py
```