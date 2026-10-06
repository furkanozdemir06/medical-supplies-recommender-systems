# Recommender System for a Medical Supplies Company

A data-driven recommender system built to help a medical supplies company generate incremental sales and revenue. The project covers data cleaning, exploratory analysis, and three complementary recommendation approaches: popularity-based, collaborative filtering (matrix factorization), and content-based (text similarity).

## Overview

The pipeline works on a relational e-commerce export that joins customers, orders, order items, and products. It answers two kinds of business questions:

- **Descriptive:** Which products sell the most, by units and by revenue? Which corporate customers spend the most?
- **Predictive:** Which products should we recommend to a given customer, or alongside a given product?

## Approaches

| # | Method | Type | What it does |
|---|--------|------|--------------|
| 1 | Popularity-based | Non-personalized | Ranks products by total units sold or total revenue |
| 2 | Matrix Factorization (SVD) | Collaborative filtering | Learns latent customer/product factors from purchase quantities and recommends products a customer has not bought yet |
| 3 | TF-IDF + Cosine Similarity | Content-based | Vectorizes product name and short description, then finds the most similar products |

## Workflow

1. **Data cleaning**
   - Dropped columns that were mostly empty (70%+ missing) and columns irrelevant to recommendation (SEO, shipping/logistics, payment, system metadata).
   - Dropped rows missing critical IDs or item cost.
   - Imputed remaining gaps: placeholders for text and categorical fields, mode for customer types, cross-column fill and median for price and cost.
2. **Exploratory analysis**
   - Top 10 products by sales volume and by revenue.
   - Order distribution for corporate (B2B) vs. individual (B2C) customers.
   - Top corporate customers by total spend, units, and number of orders.
3. **Modeling**
   - Popularity-based recommender with a selectable metric (`total_volume` or `total_revenue`).
   - SVD (`n_factors=50`) trained on a customer–product quantity matrix with an 80/20 train/test split and evaluated with RMSE.
   - Content-based recommender using TF-IDF (English stop words removed) and cosine similarity.

## Tech Stack

- Python
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn (TF-IDF, cosine similarity)
- scikit-surprise (SVD)

## Usage

```python
# 1. Popularity-based: top 5 products by units sold or revenue
popularity_based_recommender(df, top_n=5, metric="total_volume")
popularity_based_recommender(df, top_n=5, metric="total_revenue")

# 2. Personalized (SVD): top 5 unseen products for a customer
get_svd_recommendations(user_id, top_n=5)

# 3. Content-based: top 5 products similar to a given product
recommend_similar_products_content(product_id, top_n=5)
```

## Author

Furkan
