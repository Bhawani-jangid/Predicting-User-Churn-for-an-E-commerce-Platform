# ML Project: User Churn Prediction

## **Project Overview**
This project aims to predict user churn using engineered features derived from user activity logs. The project involves multiple stages, including dataset preprocessing, exploratory data analysis (EDA), feature engineering, predictive modeling, and interpretability. The following sections outline the completed work and upcoming tasks.

---

## **Completed Work**

### **1. Dataset Preprocessing**
- **Strengths:**
  - Missing values in `category_code`, `brand`, and `user_session` were handled appropriately by replacing them with "Unknown".
  - Duplicates were identified and removed.
  - `event_time` was correctly converted to a datetime format for further analysis.

- **Improvements Implemented:**
  - **Data Validation:** Checked for invalid or extreme outliers in the `price` column.
  - **Column Standardization:** Ensured all categorical columns are consistently lowercased.
  - **Enhanced Documentation:** Added comments explaining why missing values were handled in a specific way.

### **2. Exploratory Data Analysis (EDA)**
- **Strengths:**
  - Distribution analysis for `event_type`, `brand`, and `price` was visualized clearly.
  - Time-based insights (daily and hourly event distribution) provided an understanding of user activity.

- **Improvements Implemented:**
  - Analyzed user behavior over time.
  - Enhanced transition analysis using tools like pandas crosstabs.
  - Highlighted users with high activity or spending.

### **3. Feature Engineering**
- **Features Created:**
  - **Recency:** Days since the user's last activity.
  - **Frequency:** Total number of events by the user.
  - **Monetary:** Total spending by the user (for purchase events).
  - **Behavioral Metrics:** View-to-cart ratio, cart-to-purchase ratio.
  - **Session Metrics:** Number of sessions, average session duration.
  - **Categorical Preferences:** Favorite brand and category.
  - **Seasonality Features:** Extracted month and weekday from `event_time`.
  - **Interaction Terms:** Added combinations of key features to capture complex patterns.

- **Normalization:** Numerical features like `recency`, `frequency`, and `monetary` were scaled using MinMaxScaler for model compatibility.

### **4. Predictive Modeling**
- **Steps Implemented:**
  - Splitted the data into training and test sets (80/20 split).
  - Trained a Random Forest Classifier to predict user churn.
  - Evaluated the model using accuracy, classification report, and confusion matrix.

- **Results:**
  - Achieved a reasonable baseline accuracy with insightful feature importance visualization.

### **5. Interpretability**
- **Feature Importance:**
  - Highlighted key features influencing the model's decisions using Random Forest's feature importance.

- **SHAP Analysis:**
  - Used SHAP to explain individual predictions and global model behavior.
  - Generated summary plots and force plots for interpretability.

- **Partial Dependence Plots:**
  - Visualized relationships between specific features and model predictions.

---

## **Upcoming Work**

1. **Hyperparameter Tuning:**
   - Optimize the Random Forest model using GridSearchCV or RandomizedSearchCV to improve performance.

2. **Model Comparison:**
   - Experiment with other algorithms like Logistic Regression, XGBoost, and Neural Networks to identify the best-performing model.

3. **Deployment:**
   - Create a user-friendly interface (e.g., Flask or Streamlit) to allow real-time predictions.

4. **Advanced Insights:**
   - Implement clustering techniques to segment users based on behavior.
   - Provide actionable insights for business strategies.

5. **Dashboard Creation:**
   - Use visualization tools like Plotly or Tableau to create interactive dashboards for presenting results.

---

## **How to Run**
1. Clone the repository and ensure all dependencies are installed.
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   pip install -r requirements.txt
   ```
2. Place the dataset (`cleaned_events.csv`) in the project directory.
3. Run the Jupyter Notebook files in sequence:
   - `01_preprocessing.ipynb`
   - `02_eda.ipynb`
   - `03_feature_engineering.ipynb`
   - `04_modeling.ipynb`
4. Explore interpretability results in `05_interpretability.ipynb`.

---

## **Contact**
For any questions or suggestions, please feel free to contact the project maintainer.

