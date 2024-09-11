import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from keras import models
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import warnings
from flask_cors import CORS

# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", message="This is a development server.")

app = Flask(__name__)

# Enable CORS for the Flask app
CORS(app)

# MongoDB setup (Ensure the MongoDB URI is securely stored)
MONGODB_URI = os.getenv('MONGODB_URI', "mongodb+srv://adicodess11:acche4log12se13@cluster0.bjc8a.mongodb.net/AutoProposalAI")
client = MongoClient(MONGODB_URI)
db = client['AutoProposalAI']
customer_collection = db['customerrequirementinputs']
recommendation_collection = db['recommendationresults']

# Load dataset (Ensure this path is correct for your environment)
dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.csv')

# Load the dataset
try:
    car_data = pd.read_csv(dataset_path)
    car_data.columns = car_data.columns.str.strip()  # Clean column names
    car_data.fillna({'Fuel Type': 'Unknown', 'Body Type': 'Unknown', 'Transmission Type': 'Unknown'}, inplace=True)  # Handle missing values
    car_data.fillna(0, inplace=True)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please check the path.")

# Define features
selected_features = ['Ex-Showroom Price', 'Fuel Type', 'Body Type', 'Transmission Type']
numeric_features = ['Ex-Showroom Price']
categorical_features = ['Fuel Type', 'Body Type', 'Transmission Type']

# Initialize preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse_output=False), categorical_features)
    ]
)

scaler = preprocessor
scaler.fit(car_data[selected_features])

# Define and register the sampling function for VAE
@tf.keras.utils.register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Load VAE models with custom_objects
encoder = models.load_model('vae_encoder.h5', custom_objects={'sampling': sampling}, compile=False)
decoder = models.load_model('vae_decoder.h5', custom_objects={'sampling': sampling}, compile=False)

@app.route('/')
def index():
    return "Welcome to the Car Recommendation System API!"

def get_closest_cars(decoded_cars, filtered_data, n=5):
    closest_cars = []
    for _, decoded_row in decoded_cars.iterrows():
        decoded_numeric = decoded_row[numeric_features].astype(float)
        filtered_numeric = filtered_data[numeric_features].astype(float)
        distances = np.sqrt(np.sum((filtered_numeric - decoded_numeric) ** 2, axis=1))
        closest_car_indices = np.argsort(distances)[:n]
        closest_cars.extend(filtered_data.iloc[closest_car_indices].to_dict(orient='records'))
    return pd.DataFrame(closest_cars).drop_duplicates()

@app.route('/recommendations', methods=['POST'])
def recommendations():
    # Get the requirement ID from the incoming request
    data = request.json
    requirement_id = data.get('_id')  # Expecting the '_id' as the identifier

    if not requirement_id:
        return jsonify({"error": "Requirement ID is missing"}), 400

    # Fetch customer input from MongoDB based on requirement ID
    try:
        customer_input = customer_collection.find_one({"_id": ObjectId(requirement_id)})
    except:
        return jsonify({"error": "Invalid Requirement ID format."}), 400

    if not customer_input:
        return jsonify({"error": "Customer input not found for this requirement."}), 404

    # Extract the customer preferences from MongoDB data
    budget_min = customer_input.get('budgetMin', 0)
    budget_max = customer_input.get('budgetMax', float('inf'))
    body_style = customer_input.get('bodyStyle', [])
    fuel_type = customer_input.get('fuelType', 'Unknown')
    transmission_type = customer_input.get('transmissionType', 'Unknown')

    # Debugging: Print the filtering criteria
    print(f"Filtering by budgetMin: {budget_min}, budgetMax: {budget_max}, bodyStyle: {body_style}, fuelType: {fuel_type}, transmissionType: {transmission_type}")

    # Filter the dataset based on preferences and the min-max budget
    filtered_cars = car_data[
        (car_data['Ex-Showroom Price'] >= budget_min) &
        (car_data['Ex-Showroom Price'] <= budget_max) &
        (car_data['Fuel Type'] == fuel_type) &
        (car_data['Transmission Type'] == transmission_type)
    ]

    # Check if at least one of the user's body styles matches any car's body style in the dataset
    filtered_cars = filtered_cars[filtered_cars['Body Type'].apply(lambda x: any(bs in x for bs in body_style))]

    if filtered_cars.empty:
        return jsonify({"recommendations": [], "message": "No cars found matching your preferences."}), 200

    # Scale the filtered data
    x_test_scaled = scaler.transform(filtered_cars[selected_features])

    # Encode using the VAE encoder
    z_mean, z_log_var, z = encoder.predict(x_test_scaled)

    # Decode using the VAE decoder
    x_decoded = decoder.predict(z)

    # Create DataFrame with decoded data
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_feature_names)
    decoded_cars = pd.DataFrame(x_decoded, columns=feature_names)

    # Reverse the scaling for numeric features
    decoded_cars[numeric_features] = preprocessor.named_transformers_['num'].inverse_transform(decoded_cars[numeric_features])

    # Reverse one-hot encoding for categorical features
    decoded_cars[categorical_features] = preprocessor.named_transformers_['cat'].inverse_transform(decoded_cars[cat_feature_names])

    # Get the closest matches from the filtered car data
    top_recommendations = get_closest_cars(decoded_cars, filtered_cars, n=20)

    # Fetch existing recommendations (if any) to count the current number
    existing_recommendations = recommendation_collection.find_one({"createdBy": customer_input['createdBy']})
    current_recommendation_count = len(existing_recommendations['recommendations']) if existing_recommendations else 0

    # Structure to store all recommendations under a single document for the user
    recommendations_to_add = [
        {
            "recommendationId": f"recommendation{current_recommendation_count + idx + 1}",
            "Car Model": car.get('Model', ''),
            "Version": car.get('Version', ''),
            "Ex-Showroom Price": car.get('Ex-Showroom Price', 0),
            "Fuel Type": car.get('Fuel Type', ''),
            "Body Type": car.get('Body Type', ''),
            "Transmission Type": car.get('Transmission Type', '')
        }
        for idx, car in enumerate(top_recommendations.to_dict(orient='records'))
    ]

    if existing_recommendations:
        recommendation_collection.update_one(
            {"createdBy": customer_input['createdBy']},
            {"$set": {
                "recommendations": existing_recommendations['recommendations'] + recommendations_to_add,
                "updatedAt": datetime.utcnow()
            }}
        )
    else:
        recommendation_collection.insert_one({
            "createdBy": customer_input['createdBy'],
            "recommendations": recommendations_to_add,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
        })

    return jsonify({"message": "Recommendations generated and stored successfully."}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
