import streamlit as st
import pickle
import pandas as pd

# Load the trained model and the scaler
model = pickle.load(open("random_forrest_final.pkl", "rb"))
scaler = pickle.load(open("standart_scaler_final.plk", "rb"))

# Streamlit UI setup
st.sidebar.title("Loan Default Prediction")
html_temp = """
<div style="background-color:blue;padding:10px">
<h2 style="color:white;text-align:center;">Loan Default Prediction App</h2>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)

# Input fields for loan features
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0)
upfront_charges = st.sidebar.number_input("Upfront Charges", min_value=0.0)
term = st.sidebar.number_input("Term (months)", min_value=0)
property_value = st.sidebar.number_input("Property Value", min_value=0.0)
income = st.sidebar.number_input("Income", min_value=0.0)
credit_score = st.sidebar.number_input("Credit Score", min_value=0)
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
ltv = st.sidebar.number_input("Loan to Value (LTV)", min_value=0.0)
dtir1 = st.sidebar.number_input("Debt to Income Ratio (DTIR1)", min_value=0.0)

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
df = pd.get_dummies(df, columns=["age"], dtype=float)

# Ensure all necessary age categories are present
all_age_categories = [
    "age_0",
    "age_1",
    "age_2",
    "age_3",
    "age_4",
    "age_5",
    "age_6",
]
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
df[
    [
        "upfront_charges",
        "term",
        "income",
        "credit_score",
        "ltv",
        "dtir1",
        "combined_loan_property"
    ]
] = scaler.transform(
    df[
        [
            "upfront_charges",
            "term",
            "income",
            "credit_score",
            "ltv",
            "dtir1",
            "combined_loan_property"
        ]
    ]
)

# Display the loan data
st.header("Loan Information:")
st.table(df)

# Prediction button
st.subheader("Press Predict to assess the loan default risk")
if st.button("Predict"):
    prediction = model.predict(df)
    if prediction[0] == 1:
        st.error("Loan Default Risk: High")
    else:
        st.success("Loan Default Risk: Low")