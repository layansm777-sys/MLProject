import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the model and scaler
# Ensure model.pkl and scaler.pkl are in the same folder as this file
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Review Influence Predictor", layout="centered")

st.title("Shopping Reviews Influence Predictor")
st.write("This app predicts how online reviews impact your purchasing decisions using your trained SVM model.")

# --- INPUT SECTIONS ---
st.header("1. Demographics")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("What is your age?", 1, 100, 25)
with col2:
    gender = st.selectbox("What is your gender?", ["Female", "Male"])

st.header("2. Shopping Habits")
read_rev = st.selectbox("Do you usually read online reviews before making a purchase?", ["Yes", "No"])
frequency = st.selectbox("How often do you rely on online reviews when shopping?", 
                         ["Always", "Often", "Sometimes", "Rarely", "Never"])

st.header("3. Trust & Ratings")
trust = st.slider("How much do you trust online reviews compared to personal recommendations?", 1, 5, 3)
importance = st.slider("How important are star ratings in your decision-making process?", 1, 5, 3)

st.header("4. Review Content")
images = st.selectbox("Does the presence of customer-uploaded images impact your decision?", ["Yes", "No"])
videos = st.selectbox("Does the presence of customer-uploaded videos impact your decision?", ["Yes", "No"])
sentiment = st.selectbox("How does the overall sentiment (positive/negative) affect your perception?", 
                         ["Very Positive", "Positive", "Neutral", "Negative", "Very Negative"])

st.header("5. Influence Factors")
avoid_neg = st.selectbox("I avoid buying products that have negative reviews", ["Yes", "No"])
trust_users = st.selectbox("I trust user reviews more than the product description", ["Yes", "No"])
helpfulness = st.selectbox("Do you find 'Helpful' votes on reviews useful?", ["Yes", "No"])

st.header("6. Review Specifics")
recommends = st.selectbox("How likely are you to recommend a product based on positive reviews?", 
                         ["Very Likely", "Likely", "Neutral", "Unlikely", "Very Unlikely"])
many_revs = st.selectbox("Does the total number of reviews (quantity) affect your trust?", ["Yes", "No"])
recent_revs = st.selectbox("Do you specifically look for 'Recent' reviews?", ["Yes", "No"])

# --- PREDICTION LOGIC ---
if st.button("Predict Purchasing Impact"):
    # Convert text inputs to numbers
    g_num = 1 if gender == "Male" else 0
    read_num = 1 if read_rev == "Yes" else 0
    img_num = 1 if images == "Yes" else 0
    vid_num = 1 if videos == "Yes" else 0
    avoid_num = 1 if avoid_neg == "Yes" else 0
    trust_u_num = 1 if trust_users == "Yes" else 0
    help_num = 1 if helpfulness == "Yes" else 0
    many_num = 1 if many_revs == "Yes" else 0
    recent_num = 1 if recent_revs == "Yes" else 0
    
    # Mappings for sliders/scales
    freq_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4}
    sent_map = {"Very Negative": 0, "Negative": 1, "Neutral": 2, "Positive": 3, "Very Positive": 4}
    rec_map = {"Very Unlikely": 0, "Unlikely": 1, "Neutral": 2, "Likely": 3, "Very Likely": 4}

    # 4. Construct Array (14 Features Total)
    # Order matches your X columns: Age, Gender, Read, Trust, Freq, Importance, Images, Videos, Sentiment, Avoid_Neg, Trust_Users, Helpfulness, Recommends, Many_Revs, Recent_Revs
    # Note: If your model had exactly 14, we use the first 14 variables listed below.
    raw_features = np.array([[
        age, g_num, read_num, trust, freq_map[frequency], 
        importance, img_num, vid_num, sent_map[sentiment], 
        avoid_num, trust_u_num, help_num, rec_map[recommends], many_num
    ]])
    
    # 5. Scale and Predict
    try:
        scaled_features = scaler.transform(raw_features)
        prediction = model.predict(scaled_features)
        
        st.markdown("---")
        if prediction[0] == 1:
            st.success("### Result: High Influence")
            st.write("The model predicts that reviews heavily influence this user's decisions.")
        else:
            st.info("### Result: Low Influence")
            st.write("The model predicts that reviews have a minimal impact on this user.")
            
    except ValueError as e:
        st.error(f"Feature mismatch: The model expects a different number of inputs. Error: {e}")