# Athens Airbnb Price Predictor - High Performance Version
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

class AirbnbPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.features = []
        self.trained = False
        self.performance_metrics = {}
        
    def preprocess(self, data):
        df = data.copy()
        
        # Clean price 
        if 'price' in df.columns:
            df['price'] = df['price'].str.replace('[$,]', '', regex=True).astype(float)
            # More conservative outlier removal for better performance
            q01, q99 = df['price'].quantile([0.005, 0.995])
            df = df[(df['price'] >= q01) & (df['price'] <= q99) & (df['price'] > 0)]
        
        # Clean and standardize property_type column 
        if 'property_type' in df.columns:
            df['property_type'] = df['property_type'].replace({
                'Entire home/apt': 'Entire rental unit',
                'Entire serviced apartment': 'Entire rental unit',
                'Room in aparthotel': 'Room in hotel',
                'Room in boutique hotel': 'Room in hotel',
                'Private room in condo': 'Private room in rental unit',
                'Private room in home': 'Private room in rental unit',
                'Private room in serviced apartment': 'Private room in rental unit'
            })
            
            # Replace categories with less than 30 listings with 'Entire rental unit'
            prop_counts = df['property_type'].value_counts()
            small_categories = prop_counts[prop_counts < 30].index
            df.loc[df['property_type'].isin(small_categories), 'property_type'] = 'Entire rental unit'
        
        # Enhanced bathroom extraction
        if 'bathrooms_text' in df.columns:
            df['bathrooms'] = df['bathrooms_text'].str.extract(r'(\d+\.?\d*)').astype(float)
            df['bathrooms'] = df['bathrooms'].fillna(1).apply(lambda x: max(1, round(float(x))))
        
        # Precise distance calculation (km)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['distance_acropolis'] = (np.abs(df['latitude'] - 37.9715) + 
                           np.abs(df['longitude'] - 23.7267) * np.cos(np.radians(37.9715))) * 111.32
        # Critical engineered features
        if 'accommodates' in df.columns and 'bedrooms' in df.columns:
            df['room_density'] = df['accommodates'] / df['bedrooms'].replace(0, 1)
            df['space_efficiency'] = df['accommodates'] / (df['bedrooms'].replace(0, 1) + df.get('bathrooms', 1))
        
        if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
            df['bathroom_bedroom_ratio'] = df['bathrooms'] / df['bedrooms'].replace(0, 1)
        
        # Host experience (more detailed)
        if 'host_since' in df.columns:
            df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
            df['host_experience_years'] = (pd.Timestamp.now() - df['host_since']).dt.days / 365.25
            df['host_experience_years'] = df['host_experience_years'].fillna(0)
            df['is_new_host'] = (df['host_experience_years'] < 1).astype(int)
        
        # Review-based features (critical for performance)
        if 'number_of_reviews' in df.columns:
            df['review_count_log'] = np.log1p(df['number_of_reviews'].fillna(0))
            df['has_reviews'] = (df['number_of_reviews'] > 0).astype(int)
        
        if 'review_scores_rating' in df.columns:
            df['review_scores_rating'] = df['review_scores_rating'].fillna(df['review_scores_rating'].median())
            df['high_rating'] = (df['review_scores_rating'] >= 4.5).astype(int)
        
        # Availability and booking features
        if 'availability_365' in df.columns:
            df['availability_ratio'] = df['availability_365'] / 365
            df['low_availability'] = (df['availability_365'] < 90).astype(int)
        
        # Host quality features
        if 'host_is_superhost' in df.columns:
            df['is_superhost'] = (df['host_is_superhost'] == 't').astype(int)
        
        if 'instant_bookable' in df.columns:
            df['is_instant_bookable'] = (df['instant_bookable'] == 't').astype(int)
        
        if 'host_identity_verified' in df.columns:
            df['host_verified'] = (df['host_identity_verified'] == 't').astype(int)
        
        # Price per person (important feature)
        if 'accommodates' in df.columns and 'price' in df.columns:
            df['price_per_person'] = df['price'] / df['accommodates'].replace(0, 1)
        
        # Minimum nights categories
        if 'minimum_nights' in df.columns:
            df['min_nights_category'] = pd.cut(df['minimum_nights'], 
                                             bins=[0, 1, 3, 7, 30, float('inf')], 
                                             labels=[0, 1, 2, 3, 4]).astype(int)
        
        return df
    
    def train(self, csv_path='data/athens_listings.csv'):
        print("Training high-performance model...")
        
        # Load and preprocess
        df = pd.read_csv(csv_path)
        df = self.preprocess(df)
        
        # Comprehensive feature set
        numeric_features = [
            'accommodates', 'bedrooms', 'beds', 'bathrooms', 
            'minimum_nights', 'maximum_nights', 'calculated_host_listings_count',
            'distance_acropolis', 'room_density', 'bathroom_bedroom_ratio', 'space_efficiency',
            'host_experience_years', 'availability_ratio', 'is_superhost', 'is_instant_bookable',
            'review_count_log', 'has_reviews', 'review_scores_rating', 'high_rating',
            'is_new_host', 'low_availability', 'host_verified', 'min_nights_category'
        ]
        categorical_features = ['property_type', 'room_type', 'neighbourhood_cleansed']
        
        # Filter available features
        numeric_features = [f for f in numeric_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        print(f"Using {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
        
        # Create model dataset
        model_df = df[numeric_features + categorical_features + ['price']].copy()
        
        # Smart missing value handling
        for col in numeric_features:
            if col in ['accommodates', 'bedrooms', 'beds']:
                model_df[col] = model_df[col].fillna(1)
            elif col in ['bathrooms']:
                model_df[col] = model_df[col].fillna(1.0)
            elif col in ['minimum_nights']:
                model_df[col] = model_df[col].fillna(1)
            elif col in ['maximum_nights']:
                model_df[col] = model_df[col].fillna(365)
            elif col in ['calculated_host_listings_count']:
                model_df[col] = model_df[col].fillna(1)
            elif col in ['review_scores_rating']:
                model_df[col] = model_df[col].fillna(model_df[col].median())
            else:
                model_df[col] = model_df[col].fillna(model_df[col].median() if model_df[col].notna().any() else 0)
        
        # Enhanced categorical encoding (top 20 for better granularity)
        for col in categorical_features:
            model_df[col] = model_df[col].fillna('Unknown')
            top_cats = model_df[col].value_counts().head(20).index
            model_df[col] = model_df[col].apply(lambda x: x if x in top_cats else 'Other')
            
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col])
            self.encoders[col] = le
        
        # Remove rows with missing price
        model_df = model_df.dropna(subset=['price'])
        self.features = numeric_features + categorical_features
        
        # Prepare data with scaling 
        X = model_df[self.features].copy()
        y = model_df['price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features 
        numeric_indices = [i for i, col in enumerate(self.features) if col in numeric_features]
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        if numeric_indices:
            X_train_scaled.iloc[:, numeric_indices] = self.scaler.fit_transform(X_train.iloc[:, numeric_indices])
            X_test_scaled.iloc[:, numeric_indices] = self.scaler.transform(X_test.iloc[:, numeric_indices])
        
        # XGBoost parameters 
        self.model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.1,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
        
        print(f"R² Score: {r2:.3f}, RMSE: ${rmse:.2f}, MAE: ${mae:.2f}")
        
        # Store performance metrics
        self.performance_metrics = {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae
        }
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Important Features:")
        for _, row in importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        self.trained = True
        return r2, rmse, mae
    
    def predict(self, listing_data):
        if not self.trained:
            raise ValueError("Model not trained!")
        
        df = pd.DataFrame([listing_data]) if isinstance(listing_data, dict) else listing_data.copy()
        df = self.preprocess(df)
        
        # Comprehensive defaults
        defaults = {
            'accommodates': 2, 'bedrooms': 1, 'beds': 1, 'bathrooms': 1.0,
            'minimum_nights': 1, 'maximum_nights': 365, 'calculated_host_listings_count': 1,
            'distance_acropolis': 5.0, 'room_density': 2.0, 'bathroom_bedroom_ratio': 1.0, 'space_efficiency': 1.0,
            'host_experience_years': 0, 'availability_ratio': 0.8, 'is_superhost': 0, 'is_instant_bookable': 0,
            'review_count_log': 0, 'has_reviews': 0, 'review_scores_rating': 4.5, 'high_rating': 1,
            'is_new_host': 1, 'low_availability': 0, 'host_verified': 0, 'min_nights_category': 1,
            'property_type': 'Entire rental unit', 'room_type': 'Entire home/apt', 'neighbourhood_cleansed': 'Other'
        }
        
        for feature in self.features:
            if feature not in df.columns:
                df[feature] = defaults.get(feature, 0)
            elif df[feature].isna().any():
                df[feature] = df[feature].fillna(defaults.get(feature, 0))
        
        # Encode categorical features
        for col, encoder in self.encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str)
                unknown_mask = ~df[col].isin(encoder.classes_)
                if unknown_mask.any():
                    df.loc[unknown_mask, col] = encoder.classes_[0]
                df[col] = encoder.transform(df[col])
        
        # Apply same scaling as training
        X = df[self.features].copy()
        numeric_features = [f for f in self.features if f not in self.encoders]
        numeric_indices = [i for i, col in enumerate(self.features) if col in numeric_features]
        if numeric_indices:
            X.iloc[:, numeric_indices] = self.scaler.transform(X.iloc[:, numeric_indices])
        
        prediction = self.model.predict(X)
        return prediction[0] if len(prediction) == 1 else prediction
    
    def get_performance_metrics(self):
        """Return the performance metrics of the trained model"""
        return self.performance_metrics
    
    def save(self, filepath='model.pkl'):
        if self.trained:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model, 
                    'scaler': self.scaler,
                    'encoders': self.encoders, 
                    'features': self.features,
                    'performance_metrics': self.performance_metrics
                }, f)
            print(f"Model saved to {filepath}")
    
    def load(self, filepath='model.pkl'):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.encoders = data['encoders']
            self.features = data['features']
            self.performance_metrics = data.get('performance_metrics', {})
            self.trained = True
            print(f"Model loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"File {filepath} not found!")
            return False

def main():
    predictor = AirbnbPredictor()
    
    # Train model
    r2, rmse, mae = predictor.train('data/athens_listings.csv')
    predictor.save()
    
    # Test prediction
    sample = {
        'property_type': 'Apartment',
        'room_type': 'Entire home/apt',
        'accommodates': 4,
        'bedrooms': 2,
        'beds': 2,
        'bathrooms_text': '1 bath',
        'latitude': 37.9755,
        'longitude': 23.7348,
        'neighbourhood_cleansed': 'Koukaki',
        'minimum_nights': 2,
        'maximum_nights': 30,
        'availability_365': 300,
        'instant_bookable': 't',
        'host_since': '2024-01-01',
        'host_is_superhost': 'f',
        'calculated_host_listings_count': 1,
        'number_of_reviews': 15,
        'review_scores_rating': 4.8,
        'host_identity_verified': 't'
    }
    
    price = predictor.predict(sample)
    print(f"\nPredicted price: ${price:.2f}")
    print(f"Price range: ${price*0.9:.2f} - ${price*1.1:.2f}")

if __name__ == "__main__":
    main() 