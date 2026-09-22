-- 1. SCHEMA
CREATE TABLE fmcg_orders (
    order_id VARCHAR(20) PRIMARY KEY,
    order_date DATE NOT NULL,
    year SMALLINT NOT NULL,
    quarter CHAR(2) NOT NULL,
    month SMALLINT NOT NULL,
    month_name VARCHAR(12) NOT NULL,
    region VARCHAR(20) NOT NULL,
    country VARCHAR(50) NOT NULL,
    city VARCHAR(50) NOT NULL,
    sales_person VARCHAR(50) NOT NULL,
    customer_type VARCHAR(10) NOT NULL,
    sales_channel VARCHAR(20) NOT NULL, 
    promotion_type VARCHAR(30) NOT NULL,
    product_category VARCHAR(30) NOT NULL,
    brand VARCHAR(50) NOT NULL,
    product_name VARCHAR(100) NOT NULL,
    sku VARCHAR(20) NOT NULL,
    units_sold INTEGER NOT NULL,
    unit_price_usd NUMERIC(10, 2) NOT NULL,
    discount_pct NUMERIC(5, 2) NOT NULL,
    gross_sales_usd NUMERIC(12, 2) NOT NULL,
    marketing_spend_usd NUMERIC(12, 2) NOT NULL,
    cogs_usd NUMERIC(12, 2) NOT NULL,
    logistics_cost_usd NUMERIC(12, 2) NOT NULL,
    net_revenue_usd NUMERIC(12, 2) NOT NULL,
    profit_usd NUMERIC(12, 2) NOT NULL,
    profit_margin_pct NUMERIC(6, 2) NOT NULL
);

-- Indexes for common filter patterns
CREATE INDEX idx_fmcg_year_quarter ON fmcg_orders (year, quarter);
CREATE INDEX idx_fmcg_region ON fmcg_orders (region);
CREATE INDEX idx_fmcg_category ON fmcg_orders (product_category);
CREATE INDEX idx_fmcg_channel ON fmcg_orders (sales_channel);
CREATE INDEX idx_fmcg_order_date ON fmcg_orders (order_date);


-- 2. ENGINEERED FEATURES VIEW
--    Mirrors the feature engineering applied in the ML pipeline.
CREATE OR REPLACE VIEW fmcg_features AS
SELECT
    order_id,
    order_date,
    year,
    quarter,
    month,
    month_name,
    region,
    country,
    city,
    sales_person,
    customer_type,
    sales_channel,
    promotion_type,
    product_category,
    brand,
    product_name,
    sku,
    units_sold,
    unit_price_usd,
    discount_pct,
    marketing_spend_usd,
    profit_usd,
    profit_margin_pct,

    -- Effective price after discount
    ROUND(unit_price_usd * (1 - discount_pct / 100), 2) AS effective_unit_price,

    -- Estimated revenue from pre-sale variables only
    ROUND(units_sold * unit_price_usd * (1 - discount_pct / 100), 2) AS revenue_estimate,

    -- Marketing efficiency: spend per unit sold
    ROUND(marketing_spend_usd / NULLIF(units_sold, 0), 4) AS marketing_per_unit,

    -- Binary flags
    CASE WHEN promotion_type <> 'No Promo' THEN 1 ELSE 0 END AS is_promoted,
    CASE WHEN quarter = 'Q4' THEN 1 ELSE 0 END AS is_q4,

    -- Profitability flag for segmentation
    CASE WHEN profit_usd >= 0 THEN 'Profitable' ELSE 'Unprofitable' END AS profit_status

FROM fmcg_orders;


-- 3. SUMMARY VIEWS — reusable building blocks for dashboards
-- 3a. Monthly P&L summary
CREATE OR REPLACE VIEW vw_monthly_summary AS
SELECT
    year,
    month,
    month_name,
    COUNT(*) AS total_orders,
    SUM(units_sold) AS total_units,
    ROUND(SUM(gross_sales_usd), 2) AS gross_sales,
    ROUND(SUM(marketing_spend_usd), 2) AS marketing_spend,
    ROUND(SUM(profit_usd), 2) AS total_profit,
    ROUND(AVG(profit_margin_pct), 2) AS avg_margin_pct,
    SUM(CASE WHEN profit_usd < 0 THEN 1 ELSE 0 END) AS unprofitable_orders
FROM fmcg_orders
GROUP BY year, month, month_name
ORDER BY year, month;


-- 3b. Region performance
CREATE OR REPLACE VIEW vw_region_summary AS
SELECT
    region,
    COUNT(*) AS total_orders,
    ROUND(SUM(profit_usd), 2) AS total_profit,
    ROUND(AVG(profit_usd), 2) AS avg_profit_per_order,
    ROUND(AVG(profit_margin_pct), 2) AS avg_margin_pct,
    ROUND(SUM(marketing_spend_usd), 2) AS total_marketing_spend,
    ROUND(SUM(profit_usd) / NULLIF(SUM(marketing_spend_usd), 0), 4) AS marketing_roi
FROM fmcg_orders
GROUP BY region
ORDER BY total_profit DESC;


-- 3c. Category performance
CREATE OR REPLACE VIEW vw_category_summary AS
SELECT
    product_category,
    COUNT(*) AS total_orders,
    ROUND(SUM(profit_usd), 2) AS total_profit,
    ROUND(AVG(profit_usd), 2) AS avg_profit_per_order,
    ROUND(AVG(profit_margin_pct), 2) AS avg_margin_pct,
    ROUND(AVG(discount_pct), 2) AS avg_discount_pct
FROM fmcg_orders
GROUP BY product_category
ORDER BY total_profit DESC;


-- 3d. Channel performance
CREATE OR REPLACE VIEW vw_channel_summary AS
SELECT
    sales_channel,
    customer_type,
    COUNT(*) AS total_orders,
    ROUND(SUM(profit_usd), 2) AS total_profit,
    ROUND(AVG(profit_usd), 2) AS avg_profit_per_order,
    ROUND(AVG(profit_margin_pct), 2) AS avg_margin_pct
FROM fmcg_orders
GROUP BY sales_channel, customer_type
ORDER BY total_profit DESC;


-- 3e. Promotion effectiveness
CREATE OR REPLACE VIEW vw_promotion_summary AS
SELECT
    promotion_type,
    COUNT(*) AS total_orders,
    ROUND(AVG(profit_usd), 2) AS avg_profit,
    ROUND(AVG(profit_margin_pct), 2) AS avg_margin_pct,
    ROUND(AVG(discount_pct), 2) AS avg_discount_pct,
    ROUND(AVG(marketing_spend_usd), 2) AS avg_marketing_spend,
    SUM(CASE WHEN profit_usd < 0 THEN 1 ELSE 0 END) AS unprofitable_orders
FROM fmcg_orders
GROUP BY promotion_type
ORDER BY avg_profit DESC;


-- 4. ANALYTICAL QUERIES
-- 4a. Overall KPIs
SELECT
    COUNT(*) AS total_orders,
    SUM(units_sold) AS total_units_sold,
    ROUND(SUM(gross_sales_usd), 2) AS total_gross_sales,
    ROUND(SUM(profit_usd), 2) AS total_profit,
    ROUND(AVG(profit_usd), 2) AS avg_profit_per_order,
    ROUND(AVG(profit_margin_pct), 2) AS avg_profit_margin_pct,
    SUM(CASE WHEN profit_usd < 0 THEN 1 ELSE 0 END) AS unprofitable_orders,
    ROUND(
        100.0 * SUM(CASE WHEN profit_usd < 0 THEN 1 ELSE 0 END)
        / COUNT(*), 2
    ) AS unprofitable_order_pct
FROM fmcg_orders;


-- 4b. Year-over-year profit growth
SELECT
    year,
    ROUND(SUM(profit_usd), 2) AS total_profit,
    ROUND(AVG(profit_margin_pct), 2) AS avg_margin_pct,
    COUNT(*) AS total_orders,
    ROUND(
        100.0 * (SUM(profit_usd) - LAG(SUM(profit_usd)) OVER (ORDER BY year))
        / NULLIF(LAG(SUM(profit_usd)) OVER (ORDER BY year), 0),
    2) AS yoy_growth_pct
FROM fmcg_orders
GROUP BY year
ORDER BY year;


-- 4c. Quarterly seasonal pattern
SELECT
    quarter,
    ROUND(AVG(profit_usd), 2) AS avg_profit_per_order,
    ROUND(SUM(profit_usd), 2) AS total_profit,
    COUNT(*) AS total_orders
FROM fmcg_orders
GROUP BY quarter
ORDER BY quarter;


-- 4d. Top 10 most profitable products
SELECT
    product_name,
    brand,
    product_category,
    COUNT(*) AS times_ordered,
    ROUND(SUM(profit_usd), 2) AS total_profit,
    ROUND(AVG(profit_margin_pct), 2) AS avg_margin_pct
FROM fmcg_orders
GROUP BY product_name, brand, product_category
ORDER BY total_profit DESC
LIMIT 10;


-- 4e. Top 10 least profitable products (candidates for review)
SELECT
    product_name,
    brand,
    product_category,
    COUNT(*) AS times_ordered,
    ROUND(SUM(profit_usd), 2) AS total_profit,
    ROUND(AVG(profit_margin_pct), 2) AS avg_margin_pct,
    ROUND(AVG(discount_pct), 2) AS avg_discount_pct
FROM fmcg_orders
GROUP BY product_name, brand, product_category
ORDER BY total_profit ASC
LIMIT 10;


-- 4f. Marketing ROI by channel and promotion type
SELECT
    sales_channel,
    promotion_type,
    COUNT(*) AS total_orders,
    ROUND(SUM(marketing_spend_usd), 2) AS total_marketing_spend,
    ROUND(SUM(profit_usd), 2) AS total_profit,
    ROUND(
        SUM(profit_usd) / NULLIF(SUM(marketing_spend_usd), 0),
    4) AS profit_per_marketing_dollar
FROM fmcg_orders
GROUP BY sales_channel, promotion_type
ORDER BY profit_per_marketing_dollar DESC;


-- 4g. Discount sensitivity: average profit grouped by discount band
SELECT
    CASE
        WHEN discount_pct = 0 THEN '0% (No Discount)'
        WHEN discount_pct <= 5 THEN '1-5%'
        WHEN discount_pct <= 10 THEN '6-10%'
        WHEN discount_pct <= 15 THEN '11-15%'
        WHEN discount_pct <= 20 THEN '16-20%'
        ELSE '20%+'
    END AS discount_band,
    COUNT(*) AS total_orders,
    ROUND(AVG(profit_usd), 2) AS avg_profit,
    ROUND(AVG(profit_margin_pct), 2) AS avg_margin_pct,
    SUM(CASE WHEN profit_usd < 0 THEN 1 ELSE 0 END) AS unprofitable_orders
FROM fmcg_orders
GROUP BY discount_band
ORDER BY MIN(discount_pct);


-- 4h. Sales person leaderboard by total profit
SELECT
    sales_person,
    region,
    COUNT(*) AS total_orders,
    ROUND(SUM(profit_usd), 2) AS total_profit,
    ROUND(AVG(profit_usd), 2) AS avg_profit_per_order,
    ROUND(AVG(profit_margin_pct), 2) AS avg_margin_pct
FROM fmcg_orders
GROUP BY sales_person, region
ORDER BY total_profit DESC
LIMIT 20;


-- 4i. Unprofitable orders detail — for operational review
SELECT
    order_id,
    order_date,
    region,
    sales_channel,
    product_category,
    brand,
    product_name,
    units_sold,
    unit_price_usd,
    discount_pct,
    marketing_spend_usd,
    profit_usd,
    profit_margin_pct
FROM fmcg_orders
WHERE profit_usd < 0
ORDER BY profit_usd ASC;


-- 4j. Rolling 3-month average profit
SELECT
    year,
    month,
    month_name,
    ROUND(SUM(profit_usd), 2) AS monthly_profit,
    ROUND(AVG(SUM(profit_usd)) OVER (
        ORDER BY year, month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2) AS rolling_3m_avg_profit
FROM fmcg_orders
GROUP BY year, month, month_name
ORDER BY year, month;


-- 4k. Brand profitability within each category (rank)
SELECT
    product_category,
    brand,
    ROUND(SUM(profit_usd), 2) AS total_profit,
    ROUND(AVG(profit_margin_pct), 2) AS avg_margin_pct,
    RANK() OVER (
        PARTITION BY product_category
        ORDER BY SUM(profit_usd) DESC
    ) AS rank_in_category
FROM fmcg_orders
GROUP BY product_category, brand
ORDER BY product_category, rank_in_category;


-- 5. STORED PROCEDURES / HELPER QUERIES
-- 5a. Get full performance summary for a given region
-- Usage: replace 'Europe' with the desired region name
SELECT
    product_category,
    sales_channel,
    promotion_type,
    COUNT(*) AS orders,
    ROUND(SUM(profit_usd), 2) AS total_profit,
    ROUND(AVG(profit_margin_pct), 2) AS avg_margin_pct
FROM fmcg_orders
WHERE region = 'Europe'
GROUP BY product_category, sales_channel, promotion_type
ORDER BY total_profit DESC;


-- 5b. Flag high-risk orders in real time (discount > 20% with low unit price)
SELECT
    order_id,
    order_date,
    product_name,
    units_sold,
    unit_price_usd,
    discount_pct,
    marketing_spend_usd,
    profit_usd
FROM fmcg_orders
WHERE discount_pct > 20
  AND unit_price_usd < 10
ORDER BY profit_usd ASC;


-- 5c. Orders where marketing spend exceeded profit (negative ROI)
SELECT
    order_id,
    order_date,
    product_name,
    sales_channel,
    promotion_type,
    marketing_spend_usd,
    profit_usd,
    ROUND(profit_usd - marketing_spend_usd, 2) AS net_after_marketing
FROM fmcg_orders
WHERE marketing_spend_usd > profit_usd
ORDER BY net_after_marketing ASC;
