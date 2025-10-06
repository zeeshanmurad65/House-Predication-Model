# House Price Prediction Project

## 📖 Project Overview
This project uses machine learning to predict the price of houses in USD. The goal was to build and tune an advanced regression model based on house features (like area, location, and amenities) to create the most accurate price predictor possible.

---

## 📊 Dataset
The dataset used is `housePrice.csv`. It contains information about a property's features.

**Features:**
* `Area`: The total area of the house in square meters.
* `Room`: The number of bedrooms.
* `Parking`: Whether the property includes a parking space.
* `Warehouse`: Whether the property includes a warehouse/storage space.
* `Elevator`: Whether the building has an elevator.
* `Address`: The neighborhood where the property is located.
* `Price(USD)`: The target variable (**the price of the house in USD**).

---

## 🛠️ Methodology
The project followed a complete machine learning workflow:
1.  **Data Cleaning**: Handled missing values using mode imputation and removed statistical outliers from `Area` and `Price(USD)` using the IQR method.
2.  **Exploratory Data Analysis (EDA)**: Created visualizations to understand the data, such as the relationship between price and location.
3.  **Data Preprocessing**: Encoded the categorical `Address` feature using one-hot encoding and applied a **log transformation** to the `Price(USD)` target variable to stabilize its variance. All features were scaled using `StandardScaler`.
4.  **Model Training & Comparison**: Trained and evaluated several advanced regression models to find the best performer, including:
    * Gradient Boosting
    * XGBoost Regressor
    * Stacking Regressor (with Gradient Boosting and XGBoost)
5.  **Hyperparameter Tuning**: Used `GridSearchCV` to find the optimal parameters for the best-performing models to maximize the final R² score.

---

## 📈 Results
After comparing all models, the **tuned Stacking Regressor** (using a log-transformed target) provided the best performance. The process of cleaning, feature engineering, and tuning resulted in a significant improvement over the baseline model.

Here is a summary of the final model performances:

| Model | R² Score | RMSE (in USD) |
| :--- | :---: | :---: |
| **Final Tuned Stacking Model** | **~82%** | **~$39,670** |
| Tuned Gradient Boosting | ~81% | ~$41,500 |
| Baseline (Untuned) | ~67% | ~$140,000 |

The final model is not only the most accurate but also has the lowest error, meaning it is the best at predicting the actual sale price of a house.

---

## ✔️ Conclusion
The project successfully developed a model that can predict house prices with an **R² score of ~82%**. The `StackingRegressor` was identified as the most effective algorithm for this dataset, and analysis showed that the **`Area` of a house is the single most important predictive feature**.
