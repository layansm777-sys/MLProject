import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Shopping Influence Predictor", layout="centered")

st.title("Shopping Reviews Influence Predictor")
st.write("This application uses a trained Support Vector Machine (SVM) to analyze how online reviews affect purchasing behavior.")

# --- SECTION 1: Demographics ---
st.header("1. Personal Information")
col1, col2 = st.columns(2)
with col1:
    age_input = st.selectbox("What is your age range?", 
                             options=["Below 18", "18-22", "23-30", "Above 30"])
    # Mapping based on your survey image (0-3)
    age_map = {"Below 18": 0, "18-22": 1, "23-30": 2, "Above 30": 3}
    age = age_map[age_input]

with col2:
    gender_input = st.selectbox("What is your gender?", ["Female", "Male"])
    gender = 1 if gender_input == "Male" else 0

# --- SECTION 2: Shopping Habits & Platforms ---
st.header("2. Shopping Habits")
freq_input = st.selectbox("How often do you shop online?", 
                           ["Rarely", "Monthly", "Weekly", "Daily"])
freq_map = {"Rarely": 0, "Monthly": 1, "Weekly": 2, "Daily": 3}
shopping_frequency = freq_map[freq_input]

st.write("Which platforms do you usually use for shopping?")
p1, p2, p3, p4 = st.columns(4)
with p1: amazon = st.checkbox("Amazon")
with p2: noon = st.checkbox("Noon")
with p3: shein = st.checkbox("Shein")
with p4: other = st.checkbox("Other")

# --- SECTION 3: Review Engagement ---
st.header("3. Review Engagement")
read_input = st.selectbox("How often do you read reviews before buying?", 
                          ["Never", "Rarely", "Sometimes", "Often", "Always"])
read_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4}
reads_reviews = read_map[read_input]

high_ratings = st.selectbox("I prefer products that have high ratings (4 stars and above)", 
                            ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"])
rating_map = {"Strongly Disagree": 0, "Disagree": 1, "Neutral": 2, "Agree": 3, "Strongly Agree": 4}
prefer_high_ratings = rating_map[high_ratings]

# --- SECTION 4: Trust & Perception ---
st.header("4. Trust and Perception")
avoid_neg = st.selectbox("I avoid buying products that have negative reviews", 
                         ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"])
avoid_negative_reviews = rating_map[avoid_neg]

trust_u = st.selectbox("I trust user reviews more than the product description", 
                       ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"])
trust_user_reviews = rating_map[trust_u]

sentiment_input = st.selectbox(
    "Does the overall sentiment (positive/negative) of a review affect your perception?", 
    options=["No, strongly", "No", "Neutral", "Yes", "Yes, strongly"]
)
sent_map = {"No, strongly": 0, "No": 1, "Neutral": 2, "Yes": 3, "Yes, strongly": 4}
image_influence = sent_map[sentiment_input]

# --- SECTION 5: Review Quality ---
st.header("5. Review Quality")
fake_belief = st.selectbox("Do you believe some reviews are fake?", ["Yes", "Maybe", "No"])
fake_map = {"Yes": 0, "Maybe": 1, "No": 2}
fake_reviews_belief = fake_map[fake_belief]

buy_if_fake = st.selectbox("If you suspect reviews are fake, would you still buy the product?", ["Yes", "Maybe", "No"])
buy_fake_map = {"Yes": 0, "Maybe": 1, "No": 2}
buy_if_fake_reviews = buy_fake_map[buy_if_fake]

# Final details for the model
real_images = st.checkbox("Do you trust reviews more if they have real images?")
detailed_info = st.checkbox("Do you trust reviews more if they have detailed information?")

# --- PREDICTION ---
if st.button("Predict Purchasing Impact"):
    # Organize into 18 features in the exact order the Scaler expects
    # Based on your screenshots: age, gender, freq, amazon, noon, shein, other, read, pref_high, avoid_neg, trust_user, image_inf, fake_belief, buy_fake, real_img, detailed, many, recent
    raw_features = np.array([[
        age, gender, shopping_frequency, 
        1 if amazon else 0, 1 if noon else 0, 1 if shein else 0, 1 if other else 0,
        reads_reviews, prefer_high_ratings, avoid_negative_reviews, 
        trust_user_reviews, image_influence, fake_reviews_belief, 
        buy_if_fake_reviews, 1 if real_images else 0, 1 if detailed_info else 0,
        0, 0  # Placeholders for 'many_reviews' and 'recent_reviews' to reach 18
    ]])
    
    try:
        scaled_features = scaler.transform(raw_features)
        prediction = model.predict(scaled_features)
        
        st.markdown("---")
        if prediction[0] == 1:
            st.success("### Prediction: High Influence")
            st.write("This user is significantly influenced by online reviews.")
        else:
            st.info("### Prediction: Low Influence")
            st.write("This user is less likely to be swayed by online reviews.")
    except Exception as e:
        st.error(f"Error: {e}")