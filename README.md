# Retail Sales Analysis — Market Trends & Basket Insights

This project analyzes over **500,000 retail transactions** from a UK-based vendor to identify purchasing patterns, seasonality, and customer behavior. It covers the full data pipeline: from cleaning raw CSVs to generating meaningful product categories and insightful visualizations for business decision-making.

---

## Objectives

- Identify top-selling products
- Analyze seasonality: monthly, weekly, hourly trends
- Clean and standardize messy transactional data
- Aggregate repeated products in invoices
- Engineer new time-based features
- Categorize thousands of product names into 25+ business-oriented groups
- Prepare data for future market basket analysis

---

## Tools & Technologies

- **Python**  
- `Pandas`, `NumPy`, `Seaborn`, `Matplotlib`, `Plotly`
- `Swifter` (for efficient `.apply`)
- `Dash` (dashboard-ready structure)
- `Statsmodels` (normality & stationarity testing)
- `Prince` (Multiple Correspondence Analysis - MCA)
- `Scikit-learn`, `SciPy`
- `Holidays` (UK holiday detection)

---

## Data Cleaning Summary

- Removed corrupted rows and invalid transactions
- Handled missing values in product names and customer IDs
- Converted data types (e.g., price from string to float, date parsing)
- Aggregated invoice rows with the same product
- Filtered out refunds, bad debt adjustments, and zero-priced items
- Categorized thousands of items into groups using keyword rules

---

## Key Visualizations

All visualizations are available in the `images/` folder:

| Visualization                          | File                               |
|----------------------------------------|------------------------------------|
| Boxplot of Prices by Category        | `images/boxplot_prices.png`        |
| Monthly Sales Trends                 | `images/sales_trend_monthly.png`   |
| Heatmap by Weekday & Hour           | `images/heatmap_weekday_hour.png`  |
| Total Sales by Product Category     | `images/sales_by_category.png`     |
| Distribution of Quantities Sold     | `images/distribution_quantity.png` |

---

##  Project Structure


---

## Outcome

This project showcases the full data analysis workflow — from raw data cleaning to domain-specific feature engineering and compelling visual storytelling.  
It demonstrates how Python can transform transactional chaos into business clarity.

---

## Use in Portfolio

This project is part of my Upwork data science portfolio and demonstrates my ability to:
- Work with messy real-world datasets
- Build business-relevant categorizations
- Produce clean, well-documented analysis
- Communicate insights visually

Feel free to explore the code and contact me for collaborations or freelance work.
