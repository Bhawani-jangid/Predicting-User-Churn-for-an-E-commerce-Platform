import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    def __init__(self, csv_path, churn_threshold_days=30):
        """
        Initialize the ChurnPredictor with the path to events CSV file
        
        Parameters:
        csv_path: Path to the CSV file with events data
        churn_threshold_days: Number of days of inactivity to consider as churn
        """
        self.events_df = pd.read_csv(csv_path, parse_dates=['event_time'])
        self.churn_threshold_days = churn_threshold_days
        self.user_features = None
        self.model = None
        
    def preprocess_data(self):
        """Preprocess the events data for your specific format"""
        print("Starting data preprocessing...")
        
        # Handle missing values
        self.events_df['category_code'] = self.events_df['category_code'].fillna('unknown')
        self.events_df['brand'] = self.events_df['brand'].fillna('unknown')
        self.events_df['price'] = self.events_df['price'].fillna(self.events_df['price'].mean())
        
        # Sort by user_id and event_time
        self.events_df = self.events_df.sort_values(['user_id', 'event_time'])
        
        print(f"Total events: {len(self.events_df)}")
        print(f"Unique users: {self.events_df['user_id'].nunique()}")
        print(f"Date range: {self.events_df['event_time'].min()} to {self.events_df['event_time'].max()}")
        
    def create_user_features(self):
        """Create user-level features specific to your data structure"""
        print("\nCreating user features...")
        
        # Get the latest date in the dataset
        latest_date = self.events_df['event_time'].max()
        
        # Basic RFM Features
        rfm_features = self.events_df.groupby('user_id').agg({
            'event_time': [
                ('recency', lambda x: (latest_date - x.max()).days),
                ('first_purchase', lambda x: (latest_date - x.min()).days)
            ],
            'user_session': ('frequency', 'nunique'),
            'price': ('monetary', lambda x: x[self.events_df['event_type'] == 'purchase'].sum())
        })
        
        # Flatten column names
        rfm_features.columns = ['recency', 'customer_age', 'total_sessions', 'total_purchase_amount']
        
        # Event type features
        event_counts = pd.get_dummies(self.events_df['event_type']).groupby(self.events_df['user_id']).sum()
        
        # Calculate important ratios
        event_counts['view_to_cart_ratio'] = event_counts['cart'] / event_counts['view'].replace(0, 1)
        event_counts['cart_to_purchase_ratio'] = event_counts['purchase'] / event_counts['cart'].replace(0, 1)
        
        # Session features
        session_features = self.events_df.groupby(['user_id', 'user_session']).agg({
            'event_time': [
                ('session_duration', lambda x: (x.max() - x.min()).total_seconds()),
            ],
            'event_type': ('events_per_session', 'count')
        })
        
        # Aggregate session features to user level
        session_features = session_features.mean(level='user_id')
        session_features.columns = ['avg_session_duration', 'avg_events_per_session']
        
        # Price features
        price_features = self.events_df[self.events_df['event_type'] == 'purchase'].groupby('user_id').agg({
            'price': ['mean', 'std', 'max']
        }).fillna(0)
        price_features.columns = ['avg_purchase_value', 'std_purchase_value', 'max_purchase_value']
        
        # Combine all features
        self.user_features = pd.concat([
            rfm_features,
            event_counts,
            session_features,
            price_features
        ], axis=1).fillna(0)
        
        print(f"Created {len(self.user_features.columns)} features for {len(self.user_features)} users")
        
    def define_churn(self):
        """Define churn based on inactivity threshold"""
        print("\nDefining churn...")
        self.user_features['is_churned'] = (
            self.user_features['recency'] > self.churn_threshold_days
        ).astype(int)
        
        churn_rate = (self.user_features['is_churned'].mean() * 100)
        print(f"Churn rate: {churn_rate:.2f}%")
        
    def prepare_model_data(self):
        """Prepare features for modeling"""
        # Remove the churn label and any constant columns
        X = self.user_features.drop('is_churned', axis=1)
        y = self.user_features['is_churned']
        
        # Remove constant columns
        constant_columns = [col for col in X.columns if X[col].nunique() == 1]
        X = X.drop(constant_columns, axis=1)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
    def train_model(self):
        """Train the churn prediction model"""
        print("\nTraining model...")
        X_train, X_test, y_train, y_test = self.prepare_model_data()
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Print model performance
        y_pred = self.model.predict(X_test)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1]):.4f}")
        
        return X_test, y_test
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        importance_df = pd.DataFrame({
            'feature': self.user_features.drop('is_churned', axis=1).columns,
            'importance': self.model.feature_importances_
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature')
        plt.title('Top 15 Most Important Features')
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = ChurnPredictor(r"C:\Users\bhawa\Projects\interview\firstpro\events.csv", churn_threshold_days=30)
    
    # Run the pipeline
    predictor.preprocess_data()
    predictor.create_user_features()
    predictor.define_churn()
    X_test, y_test = predictor.train_model()
    predictor.plot_feature_importance()
