import streamlit as st
import pickle
import pandas as pd

# Load the trained model and the scaler

# Add debug prints
print("Starting to load model and scaler...")

# Load the trained model and the scaler
try:
    with open("random_forrest_final.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    with open("standart_scaler_final.plk", "rb") as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Error loading scaler: {e}")

print("Starting Streamlit UI setup...")
# Streamlit UI setup
st.sidebar.title("Loan Default Prediction")
html_temp = """
<div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Loan Default Prediction App</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# Input fields for loan features
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0)
upfront_charges = st.sidebar.number_input("Upfront Charges", min_value=0.0)
term = st.sidebar.number_input("Term (months)", min_value=0)
property_value = st.sidebar.number_input("Property Value", min_value=0.0)
income = st.sidebar.number_input("Income", min_value=0.0)
credit_score = st.sidebar.number_input("Credit Score", min_value=300.0, max_value=850.0, value=650.0)
age = st.sidebar.selectbox(
    "Age Range",
    (
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65-74",
        ">74",
        "<25",
    ),
)

# Calculate LTV and DTIR1
ltv = 100 * loan_amount / property_value if property_value > 0 else 0.0
dtir1 = loan_amount / income if income > 0 else 0.0

# Display calculated ratios
st.sidebar.markdown("### Calculated Ratios")
st.sidebar.metric("Loan to Value (LTV)", f"{ltv:.2f}%")
st.sidebar.metric("Debt to Income Ratio (DTIR1)", f"{dtir1:.2f}%")

# Create a dictionary with the user inputs
loan_data = {
    "loan_amount": loan_amount,
    "upfront_charges": upfront_charges,
    "term": term,
    "property_value": property_value,
    "income": income,
    "credit_score": credit_score,
    "age": age,
    "ltv": ltv,
    "dtir1": dtir1,
}

# Convert the dictionary into a Pandas DataFrame
df = pd.DataFrame.from_dict([loan_data])

# Feature engineering: combined_loan_property
df["combined_loan_property"] = df["loan_amount"] * df["property_value"]
df.drop(columns=["loan_amount", "property_value"], axis=1, inplace=True)

# One-hot encode the 'age' feature
age_mapping = {
    "<25": "age_0",
    "25-34": "age_1",
    "35-44": "age_2",
    "45-54": "age_3",
    "55-64": "age_4",
    "65-74": "age_5",
    ">74": "age_6"
}

df = pd.get_dummies(df, columns=["age"], dtype=float)

# Ensure all necessary age categories are present
all_age_categories = [f"age_{i}" for i in range(7)]
for category in all_age_categories:
    if category not in df.columns:
        df[category] = 0.0

# Reorder columns to match the model's expected input
df = df[
    [
        "upfront_charges",
        "term",
        "income",
        "credit_score",
        "ltv",
        "dtir1",
        "combined_loan_property",
        "age_0",
        "age_1",
        "age_2",
        "age_3",
        "age_4",
        "age_5",
        "age_6",
    ]
]

# Scale the numerical features
numerical_columns = [
    "upfront_charges",
    "term",
    "income",
    "credit_score",
    "ltv",
    "dtir1",
    "combined_loan_property"
]
df[numerical_columns] = scaler.transform(df[numerical_columns])

# Display the loan information
st.header("Loan Information:")
st.write("### Raw Input Data")
display_df = pd.DataFrame({
    "Metric": ["Loan Amount", "Property Value", "Income", "Term", "LTV", "DTIR1"],
    "Value": [
        f"${loan_amount:,.2f}",
        f"${property_value:,.2f}",
        f"${income:,.2f}",
        f"{term} months",
        f"{ltv:.2f}%",
        f"{dtir1:.2f}%"
    ]
})
st.table(display_df)

# Prediction button
st.subheader("Press Predict to assess the loan default risk")
if st.button("Predict"):
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
    
    if prediction[0] == 1:
        st.error(f"Loan Default Risk: High (Probability: {prediction_proba[0][1]:.2%})")
    else:
        st.success(f"Loan Default Risk: Low (Probability: {prediction_proba[0][0]:.2%})")