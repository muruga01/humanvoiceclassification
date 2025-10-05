import streamlit as st
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time

# --- Configuration ---
ASSUMED_CSV_FILE = 'vocal_gender_features_new.csv'
PCA_COMPONENTS = 10 

# Define the standardized labels for the prediction output
LABEL_MAPPING = {
    0: "Male Voice",
    1: "Female Voice",
    # Add more mappings here if you have more classes (e.g., 2: "Child Voice")
}

# --- 1. DATA AND MODEL LOADING/SIMULATION ---

@st.cache_data
def load_voice_dataset():
    try:
        # Load the CSV. Assume the first column is the index/ID and 'label' is the target.
        df = pd.read_csv(ASSUMED_CSV_FILE, index_col=0)
        
        df.columns = df.columns.str.strip()
        
        if 'label' not in df.columns:
            st.error("Error: The 'label' column (target) was not found in the CSV. Cannot proceed.")
            return pd.DataFrame(), pd.DataFrame() 

        # Separate features (X) and target (y)
        X = df.drop(columns=['label'])
        y = df['label'].astype('category').cat.codes # Encode labels numerically for the mock model

        # Standardize and make the Sample IDs uniform and sequential
        num_rows = len(X)
        new_index = [f"Sample_{i+1:03d}" for i in range(num_rows)]
        X.index = new_index
        y.index = new_index

        # Ensure all features are numeric
        if not X.apply(lambda col: pd.api.types.is_numeric_dtype(col)).all():
             st.warning("Warning: Some feature columns are not numeric. Please check your data.")

        return X, y
        
    except FileNotFoundError:
        # Fallback to synthetic data for a runnable demo if the file is missing
        st.info(f"'{ASSUMED_CSV_FILE}' not found. Generating synthetic data for demonstration.")
        n_samples = 50
        n_features = 45 # Based on your previous features list
        
        X_synth = pd.DataFrame(
            np.random.rand(n_samples, n_features) * 300, 
            columns=[f'feature_{i+1}' for i in range(n_features)]
        )
        # Create uniform IDs
        X_synth.index = [f"Sample_{i+1:03d}" for i in range(n_samples)]
        
        # Create synthetic labels (0, 1)
        y_synth = pd.Series(np.random.randint(0, 2, n_samples), index=X_synth.index)
        
        # Assign meaningful names to the first few features for visualization
        X_synth.rename(columns={0: 'mean_pitch', 1: 'std_pitch', 2: 'mfcc_1_mean'}, inplace=True)
        
        return X_synth, y_synth
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return pd.DataFrame(), pd.DataFrame()


@st.cache_resource
def load_and_simulate_pipeline(X, y):
    """
    Simulates loading the trained StandardScaler, PCA, and SVM models.
    Since we cannot load joblib files, we train a mock pipeline on the available data.
    """
    if X.empty:
        return None, None, None

    st.info(f"Simulating deployment: Training and caching the Scaler, PCA, and SVM models using {len(X)} samples.")
    
    # 1. Train the Scaler and PCA
    scaler = StandardScaler()
    pca = PCA(n_components=min(PCA_COMPONENTS, X.shape[1]))

    # We fit the scaler and PCA on the entire dataset (or the training set if split)
    # For simulation, we fit on a sample of the data.
    X_scaled = scaler.fit_transform(X)
    X_pca = pca.fit_transform(X_scaled)
    
    # 2. Train the Classifier (SVM)
    # Use a small subset for quick training simulation
    X_train, _, y_train, _ = train_test_split(X_pca, y, test_size=0.8, stratify=y, random_state=42)
    
    svm_model = SVC(kernel='linear', C=1).fit(X_train, y_train)
    
    # In a real app, you would simply load the joblib files here:
    # scaler = joblib.load('scaler.joblib')
    # pca = joblib.load('pca.joblib')
    # svm_model = joblib.load('svm_model.joblib')
    
    return scaler, pca, svm_model

def get_prediction(raw_features, scaler, pca, model, feature_names):
    """
    Runs a prediction using the full deployment pipeline:
    Raw Features -> Scaling -> PCA -> SVM Prediction.
    """
    # 1. Prepare data (must be 2D array)
    sample_df = pd.DataFrame([raw_features], columns=feature_names)
    
    time.sleep(0.5) # Simulate inference latency

    try:
        # 2. Scaling (StandardScaler)
        scaled_features = scaler.transform(sample_df)
        
        # 3. Dimensionality Reduction (PCA)
        pca_features = pca.transform(scaled_features)
        
        # 4. Final Prediction (SVM)
        prediction_code = model.predict(pca_features)[0]
        
        # Map the prediction code to the descriptive label
        predicted_label = LABEL_MAPPING.get(prediction_code, f"Predicted: Unknown Class {prediction_code}")

        return predicted_label, pca_features.flatten()
        
    except Exception as e:
        st.error(f"Prediction Pipeline Error: {e}")
        return "Prediction Failed", np.zeros(pca.n_components)


# --- 2. STREAMLIT APPLICATION UI ---

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Human Voice Classification and Clustering", layout="wide")

    # Load data (X) and target codes (y)
    X_data, y_codes = load_voice_dataset()
    
    # Load or simulate the trained pipeline components
    scaler, pca, svm_model = load_and_simulate_pipeline(X_data, y_codes)
    
    st.title("🎤 Human Voice Classification and Clustering")
    st.markdown("This app deploys the machine learning pipeline trained for real-time inference.")
    st.markdown("---")

    if X_data.empty or svm_model is None:
        return

    st.sidebar.header("Model Info")
    st.sidebar.markdown(f"- **Total Features:** {X_data.shape[1]}")
    st.sidebar.markdown(f"- **PCA Components:** {pca.n_components if pca else 'N/A'}")
    st.sidebar.markdown(f"- **Classifier:** Support Vector Machine (SVC)")
    # st.sidebar.markdown(f"---")
    # st.sidebar.markdown("This model was trained only once on app startup using synthetic data to simulate loading your artifacts.")

    # Select Box for Sample
    st.header("1. Select a Voice Sample for Prediction")
    selected_sample_id = st.selectbox(
        "Choose a Voice Sample ID:",
        X_data.index
    )

    st.markdown("---")
    
    # Get features for the selected sample
    selected_features_series = X_data.loc[selected_sample_id]
    selected_features_array = selected_features_series.to_numpy()
    
    # Prediction Button
    if st.button("🚀 Run Full Pipeline Prediction"):
        
        st.subheader(f"Input: {selected_sample_id} ({X_data.shape[1]} Raw Features)")
        
        # Display the feature data in a table
        st.dataframe(selected_features_series.to_frame('Raw Feature Value'), use_container_width=True)

        with st.spinner(f'Running prediction pipeline: Scaling -> PCA ({pca.n_components} comps) -> SVM Inference...'):
            # Run prediction using the loaded pipeline
            prediction_label, pca_scores = get_prediction(
                selected_features_array, 
                scaler, 
                pca, 
                svm_model,
                X_data.columns
            )
        
        st.markdown("## ✨ Prediction & Analysis Results")
        
        col_pred, col_pca = st.columns(2)
        
        # Prediction Card
        with col_pred:
            st.subheader("Final Classification Output")
            st.markdown(
                f"""
                <div style="background-color: #e0f7fa; padding: 20px; border-radius: 10px; border-left: 5px solid #00bcd4;">
                    <h4 style="color: #00838f; margin-top: 0;'>Model Prediction:</h4>
                    <p style="font-size: 1.5em; font-weight: bold; color: #00bcd4;">{prediction_label}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # PCA Scores Card (Visualization of the reduced features)
        with col_pca:
            st.subheader("Intermediate PCA Scores")
            pca_data = pd.DataFrame(pca_scores, index=[f'PC {i+1}' for i in range(len(pca_scores))], columns=['Score'])
            st.dataframe(pca_data, use_container_width=True, height=200)

        # Feature Visualization (The 10 Principal Components)
        st.markdown("---")
        st.subheader(f"Visualization of {pca.n_components} Principal Components")
        
        st.bar_chart(pca_data, use_container_width=True)
        st.caption("These are the 10 features used by the SVM model for final classification.")

    else:
        st.info("Click the '🚀 Run Full Pipeline Prediction' button to see the model's output and the intermediate PCA scores.")

if __name__ == "__main__":
    main()