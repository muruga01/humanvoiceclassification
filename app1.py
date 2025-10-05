import streamlit as st
import numpy as np
import pandas as pd
import time

# --- 1. DATA GENERATION AND MODEL SIMULATION ---

@st.cache_data
def generate_voice_dataset(n_samples=20):
    """Generates a synthetic dataset of voice features for demonstration."""
    np.random.seed(42)
    data = {}
    
    # Create feature names, simulating key voice analysis metrics
    feature_names = [f'MFCC_{i+1}' for i in range(7)] + ['Pitch_Mean', 'Spectral_Contrast', 'Zero_Crossing_Rate']
    
    for i in range(n_samples):
        # Generate 10 random features for each sample
        # Note: Pitch_Mean (index 7) is given a wider range for simulation purposes
        features = np.random.uniform(low=-100.0, high=100.0, size=10)
        
        # Create a unique sample name
        sample_name = f"Voice Sample {i+1:02d}"
        data[sample_name] = features
    
    df = pd.DataFrame(data).T
    df.columns = feature_names
    df.index.name = "Sample_ID"
    return df

def simulate_prediction(features):
    """
    Simulates the ML model prediction (Classification and Clustering).
    Takes a single row of features (1D numpy array).
    """
    time.sleep(0.5) # Simulate model prediction time
    
    # Simple logic for classification based on one feature (e.g., Pitch_Mean, index 7)
    # We use this to bias the random choice for a more interesting demo
    pitch_mean = features[7] 
    
    if pitch_mean > 30:
        classification = "Female Speaker (High Pitch)"
        # Bias the probability towards this classification
        p = [0.6, 0.3, 0.1]
    elif pitch_mean < -30:
        classification = "Male Speaker (Low Pitch)"
        p = [0.3, 0.6, 0.1]
    else:
        classification = "Neutral Speaker (Medium Pitch)"
        p = [0.4, 0.4, 0.2]
        
    # Classification based on simple pitch criteria (simulated)
    classification_options = [classification, "Other Gender/ID", "Uncertain"]
    final_classification = np.random.choice(classification_options, p=p)

    # Clustering (Simulating Voice Type/Group assignment)
    cluster_options = ["Cluster A: High Pitch Variability", "Cluster B: Deep Tonal Range", "Cluster C: Neutral Voiceprint"]
    clustering = np.random.choice(cluster_options)
    
    return final_classification, clustering

# --- 2. STREAMLIT APPLICATION UI ---

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Voice Analysis & Classification", layout="centered")

    # Load the synthetic dataset
    voice_data_df = generate_voice_dataset()
    
    st.title("🗣️ Human Voice Analysis & Classification")
    st.markdown("Analyze voice features from a pre-loaded synthetic dataset to demonstrate speaker classification and clustering.")
    st.markdown("---")

    # Select Box for Sample
    st.header("1. Select a Voice Sample from the Dataset")
    selected_sample_id = st.selectbox(
        "Choose a Voice Sample ID:",
        voice_data_df.index
    )

    st.markdown("---")
    
    # Get features for the selected sample
    selected_features_series = voice_data_df.loc[selected_sample_id]
    selected_features_array = selected_features_series.to_numpy()
    
    # Prediction Button
    if st.button("🚀 Run Voice Analysis on Selected Sample"):
        
        # Display features used for prediction
        st.subheader(f"Features used for **{selected_sample_id}**")
        st.dataframe(selected_features_series.to_frame('Feature Value'), use_container_width=True)

        with st.spinner(f'Analyzing features for {selected_sample_id} and running model prediction...'):
            # The selected features are passed directly to the simulation
            classification_result, clustering_result = simulate_prediction(selected_features_array)
        
        st.markdown("## ✨ Analysis Results")
        
        col_class, col_cluster = st.columns(2)
        
        # Classification Card
        with col_class:
            st.subheader("Classification (Speaker ID/Gender)")
            st.markdown(
                f"""
                <div style="background-color: #e0f7fa; padding: 20px; border-radius: 10px; border-left: 5px solid #00bcd4;">
                    <h4 style="color: #00838f; margin-top: 0;">Predicted Label:</h4>
                    <p style="font-size: 1.5em; font-weight: bold; color: #00bcd4;">{classification_result}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # Clustering Card
        with col_cluster:
            st.subheader("Clustering (Voice Type)")
            st.markdown(
                f"""
                <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; border-left: 5px solid #ff9800;">
                    <h4 style="color: #e65100; margin-top: 0;">Assigned Cluster:</h4>
                    <p style="font-size: 1.5em; font-weight: bold; color: #ff9800;">{clustering_result}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        # Feature Visualization (uses the selected sample's features)
        st.markdown("---")
        st.subheader("Feature Visualization")
        feature_data_chart = selected_features_series.to_frame('Value')
        st.bar_chart(feature_data_chart, use_container_width=True)
        st.caption(f"Bar chart showing the 10 extracted features for {selected_sample_id}.")

    else:
        st.info("Click the 'Run Voice Analysis' button after selecting a sample to see the results.")

if __name__ == "__main__":
    main()
