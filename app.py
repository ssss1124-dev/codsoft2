import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    .safe-alert {
        background: linear-gradient(135deg, #26de81 0%, #20bf6b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
</style>
""", unsafe_allow_html=True)


class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False

    def preprocess_data(self, df):
        """Preprocess the data for training/prediction"""
        df_processed = df.copy()

        # Convert trans_date_trans_time to datetime features
        df_processed['trans_date_trans_time'] = pd.to_datetime(df_processed['trans_date_trans_time'])
        df_processed['hour'] = df_processed['trans_date_trans_time'].dt.hour
        df_processed['day_of_week'] = df_processed['trans_date_trans_time'].dt.dayofweek
        df_processed['month'] = df_processed['trans_date_trans_time'].dt.month

        # Extract birth year from dob
        df_processed['dob'] = pd.to_datetime(df_processed['dob'])
        df_processed['age'] = 2019 - df_processed['dob'].dt.year

        # Create amount categories
        df_processed['amt_category'] = pd.cut(df_processed['amt'],
                                              bins=[0, 50, 100, 500, 1000, float('inf')],
                                              labels=['very_low', 'low', 'medium', 'high', 'very_high'])

        # Encode categorical variables
        categorical_columns = ['merchant', 'category', 'gender', 'city', 'state', 'job', 'amt_category']

        for col in categorical_columns:
            if col in df_processed.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        df_processed[col].astype(str))
                else:
                    # Handle unseen categories during prediction
                    try:
                        df_processed[f'{col}_encoded'] = self.label_encoders[col].transform(
                            df_processed[col].astype(str))
                    except ValueError:
                        # For unseen categories, assign a default value
                        df_processed[f'{col}_encoded'] = 0

        # Select numerical features
        feature_columns = [
                              'amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long',
                              'hour', 'day_of_week', 'month', 'age'
                          ] + [f'{col}_encoded' for col in categorical_columns if col in df_processed.columns]

        # Store feature columns for later use
        self.feature_columns = [col for col in feature_columns if col in df_processed.columns]

        return df_processed[self.feature_columns]

    def train(self, df, model_type='RandomForest'):
        """Train the fraud detection model"""
        # Preprocess data
        X = self.preprocess_data(df)
        y = df['is_fraud']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        if model_type == 'RandomForest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        else:
            self.model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)

        self.model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        self.is_trained = True
        return metrics, X_test, y_test, y_pred, y_pred_proba

    def predict(self, df):
        """Make predictions on new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X = self.preprocess_data(df)
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        return predictions, probabilities


@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    data = {
        'trans_date_trans_time': ['2019-01-01 00:00:18', '2019-01-01 00:00:44', '2019-01-01 00:00:51',
                                  '2019-01-01 00:01:16'],
        'cc_num': [2703186189652095, 630423337322, 38859492057661, 3534093764340240],
        'merchant': ['fraud_Rippin, Kub and Mann', 'fraud_Heller, Gutmann and Zieme', 'fraud_Lind-Buckridge',
                     'fraud_Kutch, Hermiston and Farrell'],
        'category': ['misc_net', 'grocery_pos', 'entertainment', 'gas_transport'],
        'amt': [4.97, 107.23, 220.11, 45.0],
        'first': ['Jennifer', 'Stephanie', 'Edward', 'Jeremy'],
        'last': ['Banks', 'Gill', 'Sanchez', 'White'],
        'gender': ['F', 'F', 'M', 'M'],
        'street': ['561 Perry Cove', '43039 Riley Greens Suite 393', '594 White Dale Suite 530',
                   '9443 Cynthia Court Apt. 038'],
        'city': ['Moravian Falls', 'Orient', 'Malad City', 'Boulder'],
        'state': ['NC', 'WA', 'ID', 'MT'],
        'zip': [28654, 99160, 83252, 59632],
        'lat': [36.0788, 48.8878, 42.1808, 46.2306],
        'long': [-81.1781, -118.2105, -112.262, -112.1138],
        'city_pop': [3495, 149, 4154, 1939],
        'job': ['Psychologist, counselling', 'Special educational needs teacher', 'Nature conservation officer',
                'Patent attorney'],
        'dob': ['1988-03-09', '1978-06-21', '1962-01-19', '1967-01-12'],
        'trans_num': ['0b242abb623afc578575680df30655b9', '1f76529f8574734946361c461b024d99',
                      'a1a22d70485983eac12b5b88dad1cf95', '6b849c168bdad6f867558c3793159a81'],
        'unix_time': [1325376018, 1325376044, 1325376051, 1325376076],
        'merch_lat': [36.011293, 49.159047, 43.150704, 47.034331],
        'merch_long': [-82.048315, -118.186462, -112.154481, -112.561071],
        'is_fraud': [0, 0, 0, 0]
    }
    return pd.DataFrame(data)


def main():
    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)

    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = FraudDetectionModel()
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None

    # Sidebar
    st.sidebar.title("🎛️ Control Panel")

    # Navigation
    page = st.sidebar.selectbox("Navigate", ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "🔍 Fraud Detection",
                                             "📈 Model Performance"])

    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Analysis":
        show_data_analysis()
    elif page == "🤖 Model Training":
        show_model_training()
    elif page == "🔍 Fraud Detection":
        show_fraud_detection()
    elif page == "📈 Model Performance":
        show_model_performance()


def show_home_page():
    st.markdown("""
    ## Welcome to the Credit Card Fraud Detection System! 🎯

    This advanced machine learning application helps detect fraudulent credit card transactions using state-of-the-art algorithms.

    ### 🌟 Features:
    - **Real-time Fraud Detection**: Instantly analyze transactions for fraud patterns
    - **Multiple ML Models**: Choose between Random Forest and Logistic Regression
    - **Interactive Visualizations**: Comprehensive data analysis and model performance metrics
    - **User-friendly Interface**: Easy-to-use Streamlit interface
    - **Detailed Analytics**: In-depth performance analysis and confusion matrices

    ### 🚀 How to Use:
    1. **📊 Data Analysis**: Explore and understand your transaction data
    2. **🤖 Model Training**: Train machine learning models on your data
    3. **🔍 Fraud Detection**: Make real-time predictions on new transactions
    4. **📈 Model Performance**: Analyze model accuracy and performance metrics

    ### 💡 Quick Start:
    Use the navigation panel on the left to explore different features. Start with uploading your data or use our sample dataset to test the system!
    """)

    # Quick stats cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🛡️ Security</h3>
            <p>Advanced ML algorithms for fraud detection</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Speed</h3>
            <p>Real-time transaction analysis</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Accuracy</h3>
            <p>High precision fraud detection</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Insights</h3>
            <p>Detailed analytics and reporting</p>
        </div>
        """, unsafe_allow_html=True)


def show_data_analysis():
    st.markdown("## 📊 Data Analysis Dashboard")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

    if st.button("Use Sample Data"):
        df = load_sample_data()
        st.success("Sample data loaded successfully!")
    elif uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully!")
    else:
        st.info("Please upload a CSV file or use sample data to continue.")
        return

    # Display basic info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Transactions", len(df))
    with col2:
        fraud_count = df['is_fraud'].sum() if 'is_fraud' in df.columns else 0
        st.metric("Fraudulent Transactions", fraud_count)
    with col3:
        fraud_rate = (fraud_count / len(df) * 100) if len(df) > 0 else 0
        st.metric("Fraud Rate (%)", f"{fraud_rate:.2f}%")

    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10))

    # Data info
    st.subheader("ℹ️ Dataset Information")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())

    with col2:
        st.write("**Data Types:**")
        st.write(df.dtypes)

    # Visualizations
    if 'is_fraud' in df.columns:
        st.subheader("📈 Data Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            # Fraud distribution
            fraud_counts = df['is_fraud'].value_counts()
            fig = px.pie(values=fraud_counts.values, names=['Legitimate', 'Fraud'],
                         title="Transaction Distribution")
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Amount distribution
            fig = px.histogram(df, x='amt', color='is_fraud',
                               title="Transaction Amount Distribution",
                               nbins=50)
            st.plotly_chart(fig, use_container_width=True)

        # Category analysis
        if 'category' in df.columns:
            category_fraud = df.groupby('category')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
            category_fraud.columns = ['Category', 'Total_Transactions', 'Fraud_Count', 'Fraud_Rate']

            fig = px.bar(category_fraud, x='Category', y='Fraud_Rate',
                         title="Fraud Rate by Category")
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    # Store data in session state
    st.session_state.data = df


def show_model_training():
    st.markdown("## 🤖 Model Training Center")

    if 'data' not in st.session_state:
        st.warning("Please upload data in the Data Analysis section first!")
        return

    df = st.session_state.data

    # Model selection
    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox("Select Model Type", ["RandomForest", "LogisticRegression"])

    with col2:
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model... This may take a few moments."):
                try:
                    metrics, X_test, y_test, y_pred, y_pred_proba = st.session_state.model.train(df, model_type)
                    st.session_state.metrics = metrics
                    st.session_state.trained = True
                    st.session_state.test_data = (X_test, y_test, y_pred, y_pred_proba)
                    st.success("🎉 Model trained successfully!")
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")

    # Display training results
    if st.session_state.trained and st.session_state.metrics:
        st.subheader("📊 Training Results")

        metrics = st.session_state.metrics

        # Metrics display
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.3f}")

        # Confusion Matrix
        st.subheader("🎯 Confusion Matrix")
        cm = metrics['confusion_matrix']

        fig = px.imshow(cm, text_auto=True, aspect="auto",
                        title="Confusion Matrix",
                        labels=dict(x="Predicted", y="Actual"))
        fig.update_xaxis(tickvals=[0, 1], ticktext=['Legitimate', 'Fraud'])
        fig.update_yaxis(tickvals=[0, 1], ticktext=['Legitimate', 'Fraud'])
        st.plotly_chart(fig, use_container_width=True)

        # Classification Report
        st.subheader("📋 Detailed Classification Report")
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        st.dataframe(report_df.round(3))


def show_fraud_detection():
    st.markdown("## 🔍 Real-time Fraud Detection")

    if not st.session_state.trained:
        st.warning("Please train a model first in the Model Training section!")
        return

    st.subheader("🎯 Single Transaction Analysis")

    # Input form for single transaction
    with st.form("transaction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            amt = st.number_input("Transaction Amount ($)", min_value=0.01, value=100.0)
            category = st.selectbox("Category", ["grocery_pos", "entertainment", "gas_transport", "misc_net", "online",
                                                 "food_dining"])
            gender = st.selectbox("Gender", ["M", "F"])

        with col2:
            merchant = st.text_input("Merchant Name", "Sample Merchant")
            city = st.text_input("City", "Sample City")
            state = st.text_input("State", "CA")

        with col3:
            lat = st.number_input("Latitude", value=40.7128)
            long = st.number_input("Longitude", value=-74.0060)
            city_pop = st.number_input("City Population", min_value=1, value=10000)

        # Additional fields
        col1, col2 = st.columns(2)
        with col1:
            job = st.text_input("Job Title", "Software Engineer")
            dob = st.date_input("Date of Birth", value=datetime(1990, 1, 1))

        with col2:
            merch_lat = st.number_input("Merchant Latitude", value=lat)
            merch_long = st.number_input("Merchant Longitude", value=long)

        submitted = st.form_submit_button("🔍 Analyze Transaction", type="primary")

        if submitted:
            # Create transaction data
            transaction_data = {
                'trans_date_trans_time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'cc_num': [1234567890123456],
                'merchant': [merchant],
                'category': [category],
                'amt': [amt],
                'first': ['John'],
                'last': ['Doe'],
                'gender': [gender],
                'street': ['123 Main St'],
                'city': [city],
                'state': [state],
                'zip': [12345],
                'lat': [lat],
                'long': [long],
                'city_pop': [city_pop],
                'job': [job],
                'dob': [dob.strftime('%Y-%m-%d')],
                'trans_num': ['test_transaction'],
                'unix_time': [int(datetime.now().timestamp())],
                'merch_lat': [merch_lat],
                'merch_long': [merch_long],
                'is_fraud': [0]  # Placeholder
            }

            transaction_df = pd.DataFrame(transaction_data)

            try:
                predictions, probabilities = st.session_state.model.predict(transaction_df)

                # Display results
                st.subheader("🎯 Analysis Results")

                if predictions[0] == 1:
                    st.markdown(f"""
                    <div class="fraud-alert">
                        🚨 FRAUD DETECTED! 🚨<br>
                        Fraud Probability: {probabilities[0]:.2%}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-alert">
                        ✅ TRANSACTION APPEARS LEGITIMATE<br>
                        Fraud Probability: {probabilities[0]:.2%}
                    </div>
                    """, unsafe_allow_html=True)

                # Risk assessment
                risk_level = "High" if probabilities[0] > 0.7 else "Medium" if probabilities[0] > 0.3 else "Low"
                st.metric("Risk Level", risk_level)

                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probabilities[0] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fraud Probability (%)"},
                    delta={'reference': 50},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, 30], 'color': "lightgreen"},
                               {'range': [30, 70], 'color': "yellow"},
                               {'range': [70, 100], 'color': "red"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                         'thickness': 0.75, 'value': 90}}))

                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")


def show_model_performance():
    st.markdown("## 📈 Model Performance Analysis")

    if not st.session_state.trained or not st.session_state.metrics:
        st.warning("Please train a model first!")
        return

    metrics = st.session_state.metrics

    # Performance metrics overview
    st.subheader("🎯 Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🎯 Accuracy", f"{metrics['accuracy']:.1%}")
    with col2:
        st.metric("🎯 Precision", f"{metrics['precision']:.1%}")
    with col3:
        st.metric("🎯 Recall", f"{metrics['recall']:.1%}")
    with col4:
        st.metric("🎯 F1-Score", f"{metrics['f1_score']:.1%}")

    # Detailed analysis
    col1, col2 = st.columns(2)

    with col1:
        # Confusion Matrix Heatmap
        st.subheader("🔥 Confusion Matrix")
        cm = metrics['confusion_matrix']

        fig = px.imshow(cm,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="Blues",
                        title="Confusion Matrix Heatmap")
        fig.update_xaxis(tickvals=[0, 1], ticktext=['Legitimate', 'Fraud'])
        fig.update_yaxis(tickvals=[0, 1], ticktext=['Legitimate', 'Fraud'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Performance metrics bar chart
        st.subheader("📊 Metrics Comparison")
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]

        fig = px.bar(x=metric_names, y=metric_values,
                     title="Model Performance Metrics",
                     color=metric_values,
                     color_continuous_scale="viridis")
        fig.update_layout(showlegend=False)
        fig.update_yaxis(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    # Classification report
    st.subheader("📋 Detailed Classification Report")
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    st.dataframe(report_df.round(4), use_container_width=True)

    # Model insights
    st.subheader("💡 Model Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
        **Model Performance Summary:**
        - The model achieved {metrics['accuracy']:.1%} overall accuracy
        - Precision: {metrics['precision']:.1%} (of predicted frauds, how many were actual frauds)
        - Recall: {metrics['recall']:.1%} (of actual frauds, how many were detected)
        - F1-Score: {metrics['f1_score']:.1%} (harmonic mean of precision and recall)
        """)

    with col2:
        if metrics['precision'] > 0.8:
            st.success("🎉 Excellent precision! Low false positive rate.")
        elif metrics['precision'] > 0.6:
            st.warning("⚠️ Good precision, but some false positives.")
        else:
            st.error("❌ Low precision - many false positives.")

        if metrics['recall'] > 0.8:
            st.success("🎉 Excellent recall! Catching most fraudulent transactions.")
        elif metrics['recall'] > 0.6:
            st.warning("⚠️ Good recall, but missing some fraudulent transactions.")
        else:
            st.error("❌ Low recall - missing many fraudulent transactions.")


if __name__ == "__main__":
    main()
