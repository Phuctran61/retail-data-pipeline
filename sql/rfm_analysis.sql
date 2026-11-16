DROP TABLE IF EXISTS mart_customer_rfm;

CREATE TABLE mart_customer_rfm AS
WITH snapshot AS (
    -- Lấy mốc ngày đối chiếu là 1 ngày sau giao dịch gần nhất trong kho dữ liệu
    SELECT DATE(MAX(invoice_date), '+1 day') AS snapshot_date
    FROM fact_sales
),
customer_metrics AS (
    -- Tổng hợp 3 chỉ số R, F, M cho từng khách hàng từ bảng Fact & Dimension
    SELECT 
        f.customer_id,
        c.country,
        CAST(
            JULIANDAY((SELECT snapshot_date FROM snapshot)) - JULIANDAY(MAX(f.invoice_date)) 
            AS INTEGER
        ) AS recency,
        COUNT(DISTINCT f.invoice_no) AS frequency,
        ROUND(SUM(f.total_amount), 2) AS monetary
    FROM fact_sales f
    JOIN dim_customers c ON f.customer_id = c.customer_id
    GROUP BY f.customer_id, c.country
),
rfm_scores AS (
    -- Chia điểm từ 1 đến 5 bằng Window Function NTILE(5)
    SELECT 
        customer_id,
        country,
        recency,
        frequency,
        monetary,
        NTILE(5) OVER (ORDER BY recency DESC) AS r_score,
        NTILE(5) OVER (ORDER BY frequency ASC) AS f_score,
        NTILE(5) OVER (ORDER BY monetary ASC) AS m_score
    FROM customer_metrics
)
SELECT 
    customer_id,
    country,
    recency,
    frequency,
    monetary,
    r_score,
    f_score,
    m_score,
    CASE 
        WHEN r_score >= 4 AND f_score >= 4 THEN 'Champions'
        WHEN r_score >= 3 AND f_score >= 3 THEN 'Loyal Customers'
        WHEN r_score >= 4 AND f_score <= 2 THEN 'New / Potential'
        WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk'
        ELSE 'Hibernating / Lost'
    END AS rfm_segment
FROM rfm_scores;