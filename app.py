import streamlit as st
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
CSV_FILE = 'vocal_gender_features_new.csv'
PCA_COMPONENTS = 10 

# Define the standardized labels for the prediction output
LABEL_MAPPING = {
    0: "Female Voice",
    1: "Male Voice"
}

# --- 1. DATA AND MODEL LOADING---

@st.cache_data
def load_voice_dataset():
    try:
        df = pd.read_csv(CSV_FILE, index_col=0)
        
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
        st.info(f"'{CSV_FILE}' not found. Generating synthetic data for demonstration.")
        n_samples = 50
        n_features = 45
        
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
    if X.empty:
        return None, None, None, None

    # 1. Train the Scaler and PCA
    scaler = StandardScaler()
    pca = PCA(n_components=min(PCA_COMPONENTS, X.shape[1]))

    # We fit the scaler and PCA on the entire dataset (or the training set if split)
    X_scaled = scaler.fit_transform(X)
    X_pca = pca.fit_transform(X_scaled)
    
    # 2. Train the Classifier (SVM)
    X_train, _, y_train, _ = train_test_split(X_pca, y, test_size=0.8, stratify=y, random_state=42)
    svm_model = SVC(kernel='linear', C=1).fit(X_train, y_train)
    
    # 3. Load K-Means
    kmeans = joblib.load('kmeans.joblib')
    st.success("Loaded pre-trained K-Means model from 'kmeans.joblib'")
    return scaler, pca, svm_model, kmeans


def get_prediction(raw_features, scaler, pca, model, kmeans, feature_names):
    """
    Runs a prediction using the full deployment pipeline:
    Raw Features -> Scaling -> PCA -> SVM Prediction + K-Means Clustering
    """
    sample_df = pd.DataFrame([raw_features], columns=feature_names)
    
    time.sleep(0.5)  # Simulate inference latency

    try:
        # 1. Scaling
        scaled_features = scaler.transform(sample_df)
        
        # 2. PCA
        pca_features = pca.transform(scaled_features)
        
        # 3. SVM Prediction
        prediction_code = model.predict(pca_features)[0]
        predicted_label = LABEL_MAPPING.get(prediction_code, f"Predicted: Unknown Class {prediction_code}")
        
        # 4. K-Means Clustering
        cluster_label = kmeans.predict(pca_features)[0]

        return predicted_label, pca_features.flatten(), cluster_label
        
    except Exception as e:
        st.error(f"Prediction Pipeline Error: {e}")
        return "Prediction Failed", np.zeros(pca.n_components), -1


# --- 2. STREAMLIT APPLICATION UI ---

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Human Voice Classification and Clustering", layout="wide")

    # Load data (X) and target codes (y)
    X_data, y_codes = load_voice_dataset()
    
    # Load or simulate the trained pipeline components (4 values)
    scaler, pca, svm_model, kmeans = load_and_simulate_pipeline(X_data, y_codes)
    
    st.title("Human Voice Classification and Clustering")
    st.markdown("This app deploys the machine learning pipeline trained for real-time inference.")
    st.markdown("---")

    if X_data.empty or svm_model is None or kmeans is None:
        st.error("Pipeline failed to load. Check data and model files.")
        return

    # Sidebar Info
    with st.sidebar:
        st.header("Model Info")
        st.markdown(f"- **Total Features:** {X_data.shape[1]}")
        st.markdown(f"- **PCA Components:** {pca.n_components}")
        st.markdown(f"- **Classifier:** Support Vector Machine (SVC)")
        st.markdown(f"- **Clustering:** K-Means ({kmeans.n_clusters} clusters)")
        st.markdown("---")
        st.markdown("Uses pre-trained `kmeans.joblib` for clustering.")

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
    if st.button("Predict the Voice"):
        
        st.subheader(f"Input: {selected_sample_id} ({X_data.shape[1]} Raw Features)")
        st.dataframe(selected_features_series.to_frame('Raw Feature Value'), use_container_width=True)

        with st.spinner(f'Running pipeline: Scaling -> PCA ({pca.n_components} comps) -> SVM + K-Means...'):
            prediction_label, pca_scores, cluster_label = get_prediction(
                selected_features_array, 
                scaler, 
                pca, 
                svm_model,
                kmeans,
                X_data.columns
            )
        
        st.markdown("## Prediction & Analysis Results")
        
        col_pred, col_pca = st.columns(2)
        
        # Prediction Card
        with col_pred:
            st.subheader("Classification Output")
            st.markdown(
                f"""
                <div style="background-color: #e0f7fa; padding: 20px; border-radius: 10px; border-left: 5px solid #00bcd4;">
                    <h4 style="color: #00838f; margin-top: 0;">Model Prediction:</h4>
                    <p style="font-size: 1.5em; font-weight: bold; color: #00bcd4;">{prediction_label}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.markdown(f"**Assigned Cluster:** `{cluster_label}`")
        
        # PCA Scores Card
        with col_pca:
            st.subheader("Intermediate PCA Scores")
            pca_data = pd.DataFrame(pca_scores, index=[f'PC {i+1}' for i in range(len(pca_scores))], columns=['Score'])
            st.dataframe(pca_data, use_container_width=True, height=200)

        # PCA Bar Chart
        st.markdown("---")
        st.subheader(f"Visualization of {pca.n_components} Principal Components")
        st.bar_chart(pca_data, use_container_width=True)
        st.caption("These are the 10 features used by the SVM model for final classification.")

    else:
        st.info("Click the 'Predict Voice' button to see the model's output and clustering.")

    # --- Clustering Explorer ---
    st.markdown("---")
    st.header("2. Explore Clustering on Full Dataset")
    
    n_clusters_user = st.slider("Number of Clusters:", 2, 6, 2, key="cluster_slider")
    
    if st.button("Run K-Means Clustering", key="run_clustering"):
        with st.spinner("Applying K-Means to entire dataset..."):
            X_scaled = scaler.transform(X_data)
            X_pca_all = pca.transform(X_scaled)
            
            kmeans_temp = KMeans(n_clusters=n_clusters_user, random_state=42, n_init='auto')
            clusters = kmeans_temp.fit_predict(X_pca_all)
            
            from sklearn.metrics import silhouette_score
            sil_score = silhouette_score(X_pca_all, clusters)
            
            st.success(f"Silhouette Score: `{sil_score:.4f}` (higher = better separation)")

            # 2D Scatter Plot
            if pca.n_components >= 2:
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(
                    X_pca_all[:, 0], X_pca_all[:, 1],
                    c=clusters, cmap='viridis', alpha=0.7, s=50
                )
                ax.set_xlabel('PC 1')
                ax.set_ylabel('PC 2')
                ax.set_title(f'K-Means Clustering ({n_clusters_user} Clusters) on PCA Features')
                plt.colorbar(scatter, ax=ax, label='Cluster')
                st.pyplot(fig)
            else:
                st.warning("PCA has fewer than 2 components. Cannot plot 2D scatter.")

if __name__ == "__main__":
    st.cache_data.clear()
    st.cache_resource.clear()
    main()