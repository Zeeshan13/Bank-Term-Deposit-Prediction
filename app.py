import streamlit as st
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Set wide mode by default
st.set_page_config(layout="wide")

# Load the dataset
df = pd.read_csv('bank_updated.csv')

# Prepare the target variable
y = df['y'].map({'yes': 1, 'no': 0})

# Rearrange columns with age_category, balance_category, and duration_category at the top
X = df[['age_category', 'balance_category', 'duration_category'] + [col for col in df.columns if col not in ['age_category', 'balance_category', 'duration_category']]]

# Drop unnecessary columns from the dataframe
X = X.drop(['y', 'duration', 'balance', 'age'], axis=1)

# Preprocessing for categorical data
categorical_cols = X.select_dtypes(include=['object']).columns
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Perform SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

import matplotlib.pyplot as plt

# Streamlit App
# Introduction Text and User Input Sidebar
st.markdown(
    """
    <div style='background-color: #8cc697; padding: 20px; border-radius: 5px;'>
        <h2 style='text-align: center;'>Bank Deposit Prediction</h2>
        <p style='text-align: justify;'>This app predicts whether a client will subscribe to a term deposit based on various factors such as age, balance, duration, job, marital status, education, etc. Please select the relevant options from the sidebar and click the "Predict" button to see the prediction.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.title('User Input')

# Add input box for client's name
client_name = st.sidebar.text_input("Client's Name")

# Sidebar inputs
age_category = st.sidebar.selectbox("Age Category", df['age_category'].unique())
balance_category = st.sidebar.selectbox("Balance Category", df['balance_category'].unique())
duration_category = st.sidebar.selectbox("Duration Category", df['duration_category'].unique())

job = st.sidebar.selectbox("Job", X['job'].unique())
marital = st.sidebar.selectbox("Marital Status", X['marital'].unique())
education = st.sidebar.selectbox("Education", X['education'].unique())
default = st.sidebar.selectbox("Has Credit in Default", X['default'].unique())
housing = st.sidebar.selectbox("Has Housing Loan", X['housing'].unique())
loan = st.sidebar.selectbox("Has Personal Loan", X['loan'].unique())
contact = st.sidebar.selectbox("Contact Communication Type", X['contact'].unique())
month = st.sidebar.selectbox("Last Contact Month", X['month'].unique())
day = st.sidebar.slider("Last Contact Day", 1, 31, 15)
campaign = st.sidebar.slider("Number of Contacts Performed During This Campaign", 1, 63, 2)
pdays = st.sidebar.slider("Number of Days That Passed by After the Client Was Last Contacted", -1, 871, 40)
previous = st.sidebar.slider("Number of Contacts Performed Before This Campaign", 0, 275, 1)
poutcome = st.sidebar.selectbox("Outcome of the Previous Marketing Campaign", X['poutcome'].unique())

# Function to preprocess user input and make prediction
def preprocess_input(age_category, balance_category, duration_category, job, marital, education, default, housing, loan, contact, month, day, campaign, pdays, previous, poutcome):
    input_data = pd.DataFrame({
        'age_category': [age_category],
        'balance_category': [balance_category],
        'duration_category': [duration_category],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'day': [day],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome]
    })

    # One-hot encode categorical variables
    input_encoded = pd.get_dummies(input_data, columns=categorical_cols)

    # Align columns with original dataset
    input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)
    
    return input_encoded

# Predict function
def predict(input_features, client_name):
    prediction = model.predict(input_features)
    prediction_text = f"{client_name} will subscribe to a term deposit." if prediction[0] == 1 else f"{client_name} will not subscribe to a term deposit."
    return prediction_text

# Get top 3 important features
def top_features():
    feature_importance = pd.Series(model.feature_importances_, index=X_encoded.columns)
    top_3_features = feature_importance.nlargest(3)
    return top_3_features

# Predict button
if st.sidebar.button('Predict'):
    input_features = preprocess_input(age_category, balance_category, duration_category, job, marital, education, default, housing, loan, contact, month, day, campaign, pdays, previous, poutcome)
    prediction = predict(input_features, client_name)

    # Display prediction
    st.subheader("Prediction")
    st.markdown(
        f"""
        <div style='background-color: #8cc697; padding: 20px; border-radius: 5px;'>
            <p>{prediction}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Add space between prediction and accuracy
    st.markdown("---")

    # Display top 3 important features
    st.subheader("Top 3 Important Features")
    top_features_df = top_features()
    st.write(top_features_df)

    # Plot bar graph of top 3 features
    st.subheader("Bar Graph of Top 3 Features")
    plt.figure(figsize=(4, 2))  # Adjust the figure size
    bars = plt.barh(top_features_df.index, top_features_df.values, color='black', height=0.4)
    plt.xlabel("Importance", fontsize=5, color='white')  # Setting color for x-axis label
    plt.ylabel("Features", fontsize=5, color='white')  # Setting color for y-axis label
    plt.title("Top 3 Most Important Features", fontsize=5, color='white')  # Setting color for title
    plt.xticks(fontsize=5, color='white')  # Setting color for x-axis ticks
    plt.yticks(fontsize=5, color='white')  # Setting color for y-axis ticks
    plt.gca().set_facecolor('#8cc697')  # Setting background color
    plt.gcf().set_facecolor('#8cc697')  # Setting background color
    plt.gcf().subplots_adjust(left=0.25)  # Adjust subplot spacing
    plt.grid(axis='x', color='#8cc697')  # Set visibility of grid lines for x-axis
    plt.grid(axis='y', color='#8cc697')  # Set visibility of grid lines for y-axis
    plt.grid(False)  # Turn off grid lines
    # Adding border color to the bars
    for bar in bars:
        bar.set_edgecolor('#8cc697')

    plt.tight_layout()  # Adjust subplot parameters to give specified padding

    st.pyplot(plt)
    # Evaluate model accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display model accuracy in a box
    st.markdown(
        f"""
        <div style='background-color: #8cc697; padding: 20px; border-radius: 5px;'>
            <p>Model Accuracy: {round(accuracy, 2)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
