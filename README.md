# Predictive Restaurant Recommender

## Project Overview
This project builds a **predictive recommendation engine** that suggests which restaurants (vendors) a customer is most likely to order from, given:
- Customer location
- Restaurant (vendor) information
- Historical order data

The model predicts the probability that a customer will order from each vendor for every saved location, and outputs results in the required format:
'CID X LOC_NUM X VENDOR, target'


## Objective
To build a **recommendation model** that predicts and ranks restaurants based on customer order behavior and restaurant attributes.

## Project Pipeline

### 1. **Data Loading & Preprocessing**
- Read and clean raw training/test CSV files.
- Merge multiple datasets: `orders`, `customers`, `locations`, and `vendors`.
- Generate all possible `(customer, location, vendor)` combinations for prediction.
- Handle missing values and categorical encodings.

### 2. **Exploratory Data Analysis (EDA)**
- Analyzed order frequency, top vendors, and customer activity patterns.
- Checked data distributions for imbalance.
- Explored vendor popularity by location and customer recency behavior.

### 3. **Feature Engineering**
Created features capturing:
- **Customer behavior**: total orders, unique vendors ordered from.
- **Vendor popularity**: total orders, average ratings.
- **Recency/frequency**: last order gap, orders per week.
- **Location features**: number of nearby vendors, average distance.

### 4. **Model Building**
- Algorithm: **LightGBM (Gradient Boosted Decision Trees)**
- Problem Type: Binary classification (ordered or not).
- Validation: **GroupKFold** by customer to prevent data leakage.
- Evaluation Metrics: `Precision@K`, `MAP@K`, and `F1-score`.

### 5. **Prediction & Ranking**
- Generate probability scores for each `(CID, LOC_NUM, VENDOR)`.
- Rank vendors by probability per customer-location pair.
- Assign `target = 1` to top-K predictions.

### 6. **Submission Format**
Final output CSV format:
| **CID X LOC_NUM X VENDOR** | **target** |
| Z59FTQD X 0 X 243 | 0 |
| 0JP29SK X 0 X 243 | 0 |
| 0JP29SK X 1 X 243 | 0 |


## Tech Stack

| Category | Tools/Libraries |
|-----------|----------------|
| Programming | Python |
| Data Handling | pandas, numpy |
| Modeling | LightGBM, scikit-learn |
| Visualization | matplotlib, seaborn |
| Utilities | tqdm, zipfile, os |


## Results Summary

| Evaluation Aspect | Description |

| Preprocessing & Understanding | Well-handled missing data, merging, and formatting 
| Data Exploration | Some insightful EDA, but could include more visualizations 
| Model Building | LightGBM with ranking evaluation and good validation 


## Repository structure
Predictive_Restaurant_Recommender

- Predictive_Restaurant_Recommender.ipynb # Main Jupyter Notebook
-  Assignment_few_output_samples.csv # Final predictions in required format
- README.md # Project documentation
- data/ # Train/Test data (not included)


## Future Improvements
- Incorporate **time-based validation** for more realistic performance.
- Implement **matrix factorization** or **neural recommenders** (e.g., NCF, embeddings).
- Apply **hyperparameter tuning** (Optuna/GridSearch) for better model accuracy.
- Add **visual dashboards** for vendor-customer insights.


## Author
Lydia Dondapati

## Final Notes
This project demonstrates the full pipeline of building a **machine learning-based restaurant recommender system**, from data preparation to ranking-based predictions.  
It showcases strong understanding of feature engineering, modeling, and reproducible data science workflows.
