# FMCG Profit Predictor

A production-style machine learning pipeline that predicts order-level profit in USD across FMCG sales transactions. The dataset covers 18,240 orders across five regions and five product categories from 2023 to 2025. The target variable is `Profit_USD`, a continuous value ranging from roughly -$640 to +$2,700.

---

## Results

Nine regression models were trained and evaluated on a held-out 20% test set. The table below shows test-set performance. R² is the primary metric because it directly measures the proportion of profit variance the model explains.

| Model              | R²     | RMSE    | MAE    | CV Mean |
|--------------------|--------|---------|--------|---------|
| Gradient Boosting  | ~0.93  | ~63     | ~38    | ~0.93   |
| Random Forest      | ~0.92  | ~68     | ~41    | ~0.92   |
| Decision Tree      | ~0.88  | ~83     | ~45    | ~0.86   |
| Ridge              | ~0.72  | ~126    | ~90    | ~0.72   |
| Linear Regression  | ~0.72  | ~126    | ~90    | ~0.72   |
| Lasso              | ~0.71  | ~128    | ~91    | ~0.71   |
| KNN                | ~0.68  | ~135    | ~82    | ~0.66   |
| AdaBoost           | ~0.62  | ~148    | ~107   | ~0.61   |
| SVR                | ~0.55  | ~161    | ~103   | ~0.54   |

Actual values will vary slightly depending on the random seed and scikit-learn version. Run the training pipeline to get exact numbers for your environment.

---

## How the Pipeline Works

Training is split into four independent components, each of which saves its output to `artifacts/` so any step can be re-run without repeating earlier ones.

Data ingestion reads the raw CSV, saves a copy to `artifacts/raw.csv`, and produces an 80/20 train/test split. The split happens before any preprocessing touches the data.

Data transformation applies feature engineering to both splits, then fits a `ColumnTransformer` exclusively on the training set. The fitted preprocessor is saved to `artifacts/preprocessor.pkl`. The test set is only transformed, never used to fit anything. This is the step that prevents data leakage from the test set influencing the scaler means, variances, or encoder category maps.

Model training loads the preprocessed arrays, trains all nine candidate regressors, compares them by R² on the test set with 5-fold cross-validation on the training set, and saves the best model to `artifacts/model.pkl`.

Model evaluation loads the saved preprocessor and model, runs them against the test set, and writes a text report plus diagnostic plots to `artifacts/evaluation/`.

The predict pipeline loads both saved artifacts and accepts a `PredictInput` dataclass that maps to the raw field values a user would enter. It applies the same feature engineering as the training transformation before calling the preprocessor and model.

---

## Feature Engineering Decisions

Several columns in the raw dataset are direct arithmetic components of `Profit_USD` — specifically `Gross_Sales_USD`, `Net_Revenue_USD`, `COGS_USD`, and `Logistics_Cost_USD`. Including any of them would make the target trivially reconstructable and produce misleading R² scores near 1.0 with no real predictive value. All four are dropped, along with `Profit_Margin_Pct` (which is derived from `Profit_USD`) and the identifier columns `Order_ID`, `SKU`, and `Order_Date`.

Five new features are created from the remaining pre-sale decision variables:

`Effective_Unit_Price` is the unit price after applying the discount percentage. It captures the actual price a customer pays rather than the list price, which is more directly connected to revenue and profit.

`Revenue_Estimate` multiplies units sold by the effective unit price to produce a proxy for top-line revenue using only variables that would be known before the order is finalised.

`Marketing_Per_Unit` divides marketing spend by units sold to measure efficiency — a campaign that generates high spend per unit is a different signal from one with low spend per unit.

`Is_Promoted` collapses the seven-level promotion type into a binary flag. The important distinction for profitability is whether any promotion was active, not which specific type it was.

`Is_Q4` flags orders from the fourth quarter. EDA showed a visible profit lift in Q4 relative to other quarters, likely driven by seasonal demand.
