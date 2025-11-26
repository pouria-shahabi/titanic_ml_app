
import streamlit as st

import pandas as pd
import joblib
# ------------------------------
# Load Model (بدون cache)
# ------------------------------
def load_model():
    model_path = "pouria_app.joblib"
    return joblib.load(model_path)

model = load_model()


# ------------------------------
# UI
# ------------------------------
st.title("🚢 Titanic Survival Prediction App")
st.write("Fill the information below and click Predict.")

# User Inputs
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", 0, 100, 30)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)
fare = st.number_input("Ticket Fare", 0.0, 600.0, 32.2)
embarked = st.selectbox("Port of Embarkation (Embarked)", ["S", "C", "Q"])

# Convert to DataFrame EXACTLY like training
input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked
}])

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.success("✔ Person Survived")
    else:
        st.error("✘ Person Did NOT Survive")
