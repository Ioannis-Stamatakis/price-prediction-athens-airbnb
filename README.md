# Athens Airbnb Price Prediction Research

A comprehensive machine learning study for predicting Airbnb prices in Athens, Greece, featuring advanced feature engineering, ML-model training and comparison, and deployment through an interactive Streamlit application. https://price-prediction-athens-airbnb.streamlit.app 

### Table of Contents
1. **[ Dataset](#dataset)** 
2. **[ Data Processing](#data-processing)** 
3. **[ Feature Engineering](#feature-engineering)** 
4. **[ Feature Selection & Model Preprocessing](#feature-selection--model-preprocessing)**
5. **[ Model Comparison](#model-training)** 
6. **[ Key Insights](#key-insights)** 
7. **[ Interactive Web Application](#interactive-web-application)** 
8. **[ How to Use](#how-to-use)**  
9. **[ Project Structure](#project-structure)** 
10. **[ Future Improvements](#future-improvements)** 
11. **[ Limitations](#limitations)**

## Dataset
This dataset contains listings web-scraped from airbnb through an open-source project called Inside.Airbnb.
- **Source**: https://insideairbnb.com/ 
- **Size**: 15,000+ listings
- **Features**: Property characteristics, host information, location data, review metrics
- **Target**: Nightly rental price (EUR)

## Data Processing
Cleaned price column, removed outliers, fixed property type text issues and more dataset refinement
**Outlier Removal:**
- Conservative approach using 0.5% and 99.5% quantiles for price filtering
- Removes extreme outliers while preserving data integrity

**Missing Value Strategy:**
- **Numerical features**: Median imputation for robust handling
- **Categorical features**: Mode imputation with "Unknown" category
- **Review features**: Zero-fill for new listings without reviews

## Feature Engineering
The engineered features had a significant impact on model performance

- **Location Features**: Distance to Acropolis using manhattan calculation, neighborhood encoding
- **Property Efficiency**: Room density, space utilization ratios
- **Host Quality**: Experience, superhost status, verification metrics  
- **Review Patterns**: Rating scores, review frequency and recency
- **Booking Dynamics**: Availability patterns, booking flexibility

## Feature Selection & Model Preprocessing

### **Feature Selection**
After extensive feature engineering, the final model uses **26 key features** that demonstrated the highest predictive power:

**Numerical Features (23):**
- `accommodates`, `bedrooms`, `beds`, `bathrooms`, `minimum_nights`, `maximum_nights`
- `calculated_host_listings_count`, `distance_acropolis`, `room_density`
- `bathroom_bedroom_ratio`, `space_efficiency`, `host_experience_years`
- `availability_ratio`, `is_superhost`, `is_instant_bookable`
- `review_count_log`, `has_reviews`, `review_scores_rating`, `high_rating`
- `is_new_host`, `low_availability`, `host_verified`, `min_nights_category`

**Categorical Features (3):**
- `property_type`, `room_type`, `neighbourhood_cleansed`

### **Data Preprocessing Pipeline**

**Feature Scaling:**
- **StandardScaler** applied to all numerical features
- Ensures equal weight across features with different scales
- Critical for gradient-based algorithms

**Categorical Encoding:**
- **Label Encoding** for ordinal categorical variables
- **Top-20 category preservation** for high-cardinality features (neighborhoods)
- Rare categories grouped into "Other" to prevent overfitting

**Train/Test Split:**
- **80/20 split** with stratified sampling by neighborhood
- **Random state fixed** for reproducible results
- Separate preprocessing pipelines to prevent data leakage

## Model Training
**In the notebook I trained five different machine learning models to see which performs better on real estate data:**

| Model | R² Score | RMSE (€) | MAE (€) |
|-------|----------|----------|---------|
| **XGBoost** | **76.0%** | **€43.65** | **€24.28** |
| LightGBM | 75.9% | €44.08 | €24.90 |
| Gradient Boosting | 75.5% | €44.08 | €24.90 |
| Random Forest | 72.2% | €46.95 | €26.30 |
| Linear Regression | 61.6% | €55.16 | €35.64 |

![Model Comparison](data/output1.png)

Our comparison shows that Gradient Boosting models perform better than linear and simple random forest models.

**Model Interpretation:**
- Explains 76% of price variance
- Average prediction error: €24.28 per night
- Strong performance across all property types and neighborhoods

## Key Insights

1. **Location Impact**: Distance to major attractions (Acropolis) shows strong predictive power
2. **Host Quality**: Superhost status and experience significantly affect pricing
3. **Property Efficiency**: Room density and space efficiency are crucial factors
4. **Review Patterns**: Both quantity and quality of reviews impact pricing decisions

![Athens City](data/output.png)

## Interactive Web Application

The model is deployed through a user-friendly Streamlit interface featuring:

### **Core Features**
- **Interactive Map**: Click-to-select property location in Athens
- **Real-time Predictions**: Instant price estimates based on input parameters
- **Model Transparency**: Display of actual performance metrics

### **Technical Implementation**
- **Frontend**: Streamlit with custom CSS styling
- **Backend**: Scikit-learn pipeline with XGBoost
- **Deployment**: Streamlit Cloud

## How to Use

### **Live Web Application** 
 **[Try the Live Demo on Streamlit Cloud](https://price-prediction-athens-airbnb.streamlit.app/)**

### **Try Locally**

**Install Requirements**
```bash
pip install -r requirements.txt
```
```bash
streamlit run Athens_Price_Prediction_Streamlit.py
```

### **Research Notebook**
Explore the full model development process:
```bash
jupyter notebook Notebook/model_pipeline_notebook.ipynb
```

### **Command Line Sample Prediction**
```bash
python model/predict_my_listing_price.py
```

### **Model Training**
Retrain with updated data:
```bash
python model/airbnb_price_predictor.py
```

## Project Structure

```
Athens-Airbnb-Price-Prediction/
├── Notebook/
│   └── model_pipeline_notebook.ipynb    # Complete research workflow
├── model/
│   ├── airbnb_price_predictor.py        # Core ML pipeline
│   └── predict_my_listing_price.py      # CLI interface
├── data/
│   ├── athens_listings.csv              # Training dataset
│   └── city-5761429_1920.jpg           # UI background
├── Athens_Price_Prediction_Streamlit.py # Web application
├── model.pkl                           # Trained model 
├── requirements.txt                    # Dependencies
└── README.md                          
```


## Future Improvements

1. **Temporal Analysis**: Seasonal pricing patterns and demand forecasting
2. **Deep Learning**: Neural network architectures for complex feature interactions
3. **Multi-City Models**: Generalization across different cities
4. **Dynamic Pricing**: Real-time price optimization algorithms
5. **External Factors**: Integration of events, weather, and economic indicators

## Limitations

**The dataset provided by inside.airbnb contains listings only near the center of Athens, suburb areas are not included so the prediction of these neighboorhoods will not be accurate**








