import streamlit as st
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor


# Load model
model = joblib.load('price_predictor_model.pkl')  # Replace with your model path

# Function to predict the price
def predict_price(features):
    """
    Predict the price by applying inverse log transformation.
    """
    log_pred = model.predict([features])
    pred_price = np.exp(log_pred)  # Inverse of the log transformation
    return pred_price[0]

# Streamlit App Interface starts here
def main():
    st.title("Airbnb Price Prediction")
    st.write("We help Airbnb host set the right price for their listing.")
    
    df = pd.read_csv('listings_selected_final.csv')

    # Inputs for features
    latitude = st.number_input("Enter Latitude")
    longitude = st.number_input("Enter Longitude")
    accommodates = int(st.number_input("Accommodates how many people?"))
    bathrooms = int(st.number_input("Number of bathrooms?"))
    bedrooms = int(st.number_input("Number of bedrooms?"))
    beds = int(st.number_input("Number of beds?"))
    
    # Convert boolean checkboxes to 1 or 0
    instant_bookable_value = 1 if st.checkbox("Instant Bookable?", value=False) else 0
    pets_value = 1 if st.checkbox("Are pets allowed?", value=False) else 0
    dryer_value = 1 if st.checkbox("Is there a dryer?", value=False) else 0
    indoor_fireplace_value = 1 if st.checkbox("Is there an indoor fireplace?", value=False) else 0
    elevator_value = 1 if st.checkbox("Is there an elevator?", value=False) else 0
    pool_value = 1 if st.checkbox("Is there a pool?", value=False) else 0
    hair_dryer_value = 1 if st.checkbox("Is there a hair dryer?", value=False) else 0
    washer_value = 1 if st.checkbox("Is there a washer?", value=False) else 0
    wheelchair_value = 1 if st.checkbox("Is the place wheelchair-accessible?", value=False) else 0
    free_parking_value = 1 if st.checkbox("Is there a free parking?", value=False) else 0
    family_kid_friendly_value = 1 if st.checkbox("Is the place family/kids friendly?", value=False) else 0
    essentials_value = 1 if st.checkbox("Are there essentials provided?", value=False) else 0
    hot_tub_value = 1 if st.checkbox("Is there a hot tub?", value=False) else 0
    smoking_value = 1 if st.checkbox("Is smoking allowed?", value=False) else 0
    tv_value = 1 if st.checkbox("Is there a TV?", value=False) else 0
    internet_value = 1 if st.checkbox("Is there internet?", value=False) else 0
    first_aid_kit_value = 1 if st.checkbox("Is there a first aid kit?", value=False) else 0
    breakfast_value = 1 if st.checkbox("Is breakfast provided?", value=False) else 0
    heating_value = 1 if st.checkbox("Is there heating?", value=False) else 0
    pets_on_property_value = 1 if st.checkbox("Are there pets on property?", value=False) else 0
    all_day_checkin_value = 1 if st.checkbox("Is there a 24 hour check-in?", value=False) else 0
    laptop_value = 1 if st.checkbox("Is there a laptop?", value=False) else 0
    intercom_value = 1 if st.checkbox("Is there an intercom?", value=False) else 0
    aircon_value = 1 if st.checkbox("Is there air conditioning?", value=False) else 0
    fire_ext_value = 1 if st.checkbox("Is there a fire extinguisher?", value=False) else 0
    iron_value = 1 if st.checkbox("Is there an iron?", value=False) else 0
    cat_value = 1 if st.checkbox("Are there cats around?", value=False) else 0
    cable_TV_value = 1 if st.checkbox("Is there a cable TV?", value=False) else 0
    lock_in_room_value = 1 if st.checkbox("Is there a lock in any room?", value=False) else 0
    gym_value = 1 if st.checkbox("Is there a gym?", value=False) else 0
    doorman_value = 1 if st.checkbox("Is there a doorman?", value=False) else 0
    wireless_internet_value = 1 if st.checkbox("Is wifi provided?", value=False) else 0
    kitchen_value = 1 if st.checkbox("Is there a kitchen?", value=False) else 0
    shampoo_value = 1 if st.checkbox("Is there shampoo?", value=False) else 0
    public_transpotation_value = 1 if st.checkbox("Is there any public transportation?", value=False) else 0

    # Select categorical inputs
    selected_neighbourhood = st.selectbox("Select Neighbourhood", df['neighbourhood_cleansed'].unique())
    selected_property_type = st.selectbox("Select Property Type", df['property_type'].unique())
    selected_room_type = st.selectbox("Select Room Type", df['room_type'].unique())
    selected_bed_type = st.selectbox("Select Bed Type", df['bed_type'].unique())
    cancellation_policy_type = st.selectbox("Select Cancellation Policy", df['cancellation_policy'].unique())
    room_size = st.selectbox("Select Room Size", df['room_size'].unique())

    # One-hot encoding for the categorical features (1 if selected, 0 for all unselected columns)
    neighbourhood_one_hot = pd.get_dummies(df['neighbourhood_cleansed'], prefix='neighbourhood_cleansed')
    neighbourhood_one_hot = neighbourhood_one_hot.columns.to_series().apply(lambda x: 1 if x == f'neighbourhood_cleansed_{selected_neighbourhood}' else 0)

    property_type_one_hot = pd.get_dummies(df['property_type'], prefix='property_type')
    property_type_one_hot = property_type_one_hot.columns.to_series().apply(lambda x: 1 if x == f'property_type_{selected_property_type}' else 0)

    room_type_one_hot = pd.get_dummies(df['room_type'], prefix='room_type')
    room_type_one_hot = room_type_one_hot.columns.to_series().apply(lambda x: 1 if x == f'room_type_{selected_room_type}' else 0)

    bed_type_one_hot = pd.get_dummies(df['bed_type'], prefix='bed_type')
    bed_type_one_hot = bed_type_one_hot.columns.to_series().apply(lambda x: 1 if x == f'bed_type_{selected_bed_type}' else 0)

    cancellation_policy_one_hot = pd.get_dummies(df['cancellation_policy'], prefix='cancellation_policy')
    cancellation_policy_one_hot = cancellation_policy_one_hot.columns.to_series().apply(lambda x: 1 if x == f'cancellation_policy_{cancellation_policy_type}' else 0)

    room_size_one_hot = pd.get_dummies(df['room_size'], prefix='room_size')
    room_size_one_hot = room_size_one_hot.columns.to_series().apply(lambda x: 1 if x == f'room_size_{room_size}' else 0)

    # st.write(neighbourhood_one_hot) ##Checking purpose

    features = np.concatenate([
        np.array([latitude, longitude, accommodates, bathrooms, bedrooms, beds,
                  instant_bookable_value, pets_value, dryer_value, indoor_fireplace_value, elevator_value, 
                  pool_value, hair_dryer_value, washer_value, wheelchair_value, free_parking_value, 
                  family_kid_friendly_value, essentials_value, hot_tub_value, smoking_value, tv_value, 
                  internet_value, first_aid_kit_value, breakfast_value, heating_value, pets_on_property_value, 
                  all_day_checkin_value, laptop_value, intercom_value, aircon_value, fire_ext_value, iron_value, 
                  cat_value, cable_TV_value, lock_in_room_value, gym_value, doorman_value, wireless_internet_value, 
                  kitchen_value, shampoo_value, public_transpotation_value]),
        neighbourhood_one_hot.values,
        property_type_one_hot.values,
        room_type_one_hot.values,
        bed_type_one_hot.values,
        cancellation_policy_one_hot.values,
        room_size_one_hot.values
    ])

    # Make prediction when user clicks button
    if st.button("Predict Price"):
        predicted_price = predict_price(features)
        st.write(f"Predicted Price: ${predicted_price:.2f}")

if __name__ == '__main__':
    main()
