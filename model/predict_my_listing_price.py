# Simple Airbnb Price Predictor
from airbnb_price_predictor import AirbnbPredictor
import os

def main():
    # Sample listing - edit these values for your property
    listing = {
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
        'number_of_reviews': 0,
        'review_scores_rating': 4.5,
        'host_identity_verified': 'f'
    }
    
    try:
        # Load model
        if not os.path.exists('model.pkl'):
            print("Model not found! Run 'python airbnb_price_predictor.py' first.")
            return
        
        predictor = AirbnbPredictor()
        predictor.load('model.pkl')
        
        # Predict price
        price = predictor.predict(listing)
        
        # Display results
        print(f"Property: {listing['accommodates']} guests, {listing['bedrooms']} bed, {listing['neighbourhood_cleansed']}")
        print(f"Predicted Price: €{price:.2f}/night")
        print(f"Price Range: €{price*0.9:.2f} - €{price*1.1:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 