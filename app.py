import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from model_loader import ModelLoader

# Page configuration
st.set_page_config(
    page_title="Recipe Traffic Classifier",
    page_icon="🍳",
    layout="wide"
)

# Title and description
st.title("🍳 Recipe Traffic Classifier")
st.markdown("""
This app predicts whether a recipe will generate **High** or **Low** traffic based on its nutritional 
and categorical features. Upload a CSV file with recipe data to get predictions.
""")

# Load the model, scaler, and encoder using the dynamic loader
@st.cache_resource
def load_model():
    try:
        loader = ModelLoader(model_dir="model")
        model, scaler, encoder, metadata = loader.load_all()
        category_encoding = loader.get_category_encoding()
        return model, scaler, encoder, category_encoding, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None

model, scaler, encoder, CATEGORY_ENCODING, metadata = load_model()


def preprocess_data(df):
    """
    Preprocess the data following the exact steps from the notebook
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Drop high_traffic column if it exists (it's the target, not a feature)
    if 'high_traffic' in data.columns:
        data = data.drop('high_traffic', axis=1)
    
    # Check required columns
    required_cols = ['recipe', 'calories', 'carbohydrate', 'sugar', 'protein', 'category', 'servings']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return None
    
    # Check for missing category values - reject as we can't handle this properly
    if data['category'].isnull().any():
        st.error("Data contains missing 'category' values. Please ensure all recipes have a category assigned.")
        return None
    
    # Check for unknown categories (only if CATEGORY_ENCODING is available)
    if CATEGORY_ENCODING:
        unknown_categories = set(data['category'].unique()) - set(CATEGORY_ENCODING.keys())
        if unknown_categories:
            st.warning(f"Unknown categories found: {', '.join(unknown_categories)}. They will be encoded using the mean value.")
    
    # Handle servings column (convert to int) - as done in notebook
    if 'servings' in data.columns:
        data['servings'] = data['servings'].replace({'4 as a snack': 4, '6 as a snack': 6})
        data['servings'] = pd.to_numeric(data['servings'], errors='coerce').fillna(4).astype(int)
    
    # Handle missing values in nutritional columns
    # Fill with category-based mean (as done in the notebook)
    for col in ['calories', 'carbohydrate', 'sugar', 'protein']:
        if col in data.columns:
            # Convert to numeric first
            data[col] = pd.to_numeric(data[col], errors='coerce')
            # Fill missing values with category mean (exactly as in notebook)
            data[col] = data.groupby('category')[col].transform(
                lambda x: x.fillna(round(x.mean(), 2))
            )
            # If still any NaN (e.g., entire category is NaN), fill with overall mean
            if data[col].isnull().any():
                data[col] = data[col].fillna(round(data[col].mean(), 2))
    
    return data

def encode_categorical(data, category_encoder, category_encoding_dict):
    """
    Encode categorical variable using the fitted encoder from training.
    Falls back to manual mapping if encoder transform fails.
    """
    data = data.copy()
    if 'category' not in data.columns:
        return data
    
    try:
        # Try using the encoder's transform method
        data['category'] = category_encoder.transform(data[['category']])
    except Exception as e:
        # Fallback to manual mapping using the encoding dictionary
        if category_encoding_dict:
            mean_value = sum(category_encoding_dict.values()) / len(category_encoding_dict)
            data['category'] = data['category'].map(category_encoding_dict).fillna(mean_value)
        else:
            raise ValueError(f"Cannot encode categories: encoder transform failed and no encoding dictionary available. Error: {e}")
    
    return data

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a page:", ["Single Prediction", "Batch Prediction", "Model Info"])

# Get available categories
available_categories = list(CATEGORY_ENCODING.keys()) if CATEGORY_ENCODING else ["Unknown"]

if page == "Single Prediction":
    st.header("📝 Single Recipe Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recipe Details")
        recipe_name = st.text_input("Recipe Name", "My Recipe")
        category = st.selectbox("Category", available_categories)
        servings = st.number_input("Servings", min_value=1, max_value=20, value=4)
    
    with col2:
        st.subheader("Nutritional Information")
        calories = st.number_input("Calories", min_value=0.0, max_value=2000.0, value=200.0, step=10.0)
        carbohydrate = st.number_input("Carbohydrate (g)", min_value=0.0, max_value=200.0, value=30.0, step=1.0)
        sugar = st.number_input("Sugar (g)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
        protein = st.number_input("Protein (g)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    
    if st.button("🔮 Predict Traffic", type="primary"):
        if model and scaler and encoder:
            # Create dataframe
            input_data = pd.DataFrame({
                'recipe': [recipe_name],
                'calories': [calories],
                'carbohydrate': [carbohydrate],
                'sugar': [sugar],
                'protein': [protein],
                'category': [category],
                'servings': [servings]
            })
            
            # Preprocess
            processed_data = preprocess_data(input_data)
            
            if processed_data is not None:
                # Encode category using the encoder
                processed_data = encode_categorical(processed_data, encoder, CATEGORY_ENCODING)
                
                # Prepare features (exclude recipe name)
                X = processed_data.drop('recipe', axis=1)
                
                # Scale features
                X_scaled = scaler.transform(X)
                
                # Predict
                prediction = model.predict(X_scaled)[0]
                prediction_proba = model.predict_proba(X_scaled)[0]
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prediction", "🔥 High Traffic" if prediction == 1 else "❄️ Low Traffic")
                
                with col2:
                    st.metric("Confidence", f"{max(prediction_proba)*100:.2f}%")
                
                with col3:
                    st.metric("Recipe", recipe_name)
                
                # Probability visualization
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Low Traffic', 'High Traffic'],
                        y=[prediction_proba[0]*100, prediction_proba[1]*100],
                        marker_color=['lightblue', 'salmon']
                    )
                ])
                fig.update_layout(
                    title="Traffic Probability Distribution",
                    yaxis_title="Probability (%)",
                    xaxis_title="Traffic Class",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

elif page == "Batch Prediction":
    st.header("📊 Batch Recipe Prediction")
    
    st.markdown("""
    Upload a CSV file containing multiple recipes. The file should have the following columns:
    - `recipe`: Recipe name/ID
    - `calories`: Calorie content
    - `carbohydrate`: Carbohydrate content (g)
    - `sugar`: Sugar content (g)
    - `protein`: Protein content (g)
    - `category`: Recipe category
    - `servings`: Number of servings
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.info(f"Loaded {len(df)} recipes")
            
            if st.button("🚀 Predict All Recipes", type="primary"):
                if model and scaler and encoder:
                    with st.spinner("Processing recipes..."):
                        # Preprocess data
                        processed_df = preprocess_data(df)
                        
                        if processed_df is not None:
                            # Store recipe names for later
                            recipe_names = processed_df['recipe'].values
                            
                            # Encode category using the encoder
                            processed_df = encode_categorical(processed_df, encoder, CATEGORY_ENCODING)
                            
                            # Prepare features
                            X = processed_df.drop('recipe', axis=1)
                            
                            # Scale features
                            X_scaled = scaler.transform(X)
                            
                            # Predict
                            predictions = model.predict(X_scaled)
                            predictions_proba = model.predict_proba(X_scaled)
                            
                            # Create results dataframe
                            results_df = df.copy()
                            results_df['predicted_traffic'] = ['High' if p == 1 else 'Low' for p in predictions]
                            results_df['confidence_low'] = [f"{p[0]*100:.2f}%" for p in predictions_proba]
                            results_df['confidence_high'] = [f"{p[1]*100:.2f}%" for p in predictions_proba]
                            results_df['max_confidence'] = [f"{max(p)*100:.2f}%" for p in predictions_proba]
                            
                            st.success("✅ Predictions Complete!")
                            
                            # Summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Recipes", len(results_df))
                            
                            with col2:
                                high_count = sum(predictions)
                                st.metric("High Traffic", high_count)
                            
                            with col3:
                                low_count = len(predictions) - high_count
                                st.metric("Low Traffic", low_count)
                            
                            with col4:
                                avg_confidence = np.mean([max(p) for p in predictions_proba]) * 100
                                st.metric("Avg Confidence", f"{avg_confidence:.2f}%")
                            
                            # Display results
                            st.subheader("📈 Prediction Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Visualization
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Traffic distribution
                                fig1 = px.pie(
                                    values=[high_count, low_count],
                                    names=['High Traffic', 'Low Traffic'],
                                    title='Traffic Distribution',
                                    color_discrete_sequence=['salmon', 'lightblue']
                                )
                                st.plotly_chart(fig1, use_container_width=True)
                            
                            with col2:
                                # Category-wise traffic
                                category_traffic = results_df.groupby(['category', 'predicted_traffic']).size().reset_index(name='count')
                                fig2 = px.bar(
                                    category_traffic,
                                    x='category',
                                    y='count',
                                    color='predicted_traffic',
                                    title='Traffic Prediction by Category',
                                    color_discrete_map={'High': 'salmon', 'Low': 'lightblue'},
                                    barmode='group'
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="recipe_predictions.csv",
                                mime="text/csv"
                            )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

elif page == "Model Info":
    st.header("ℹ️ Model Information")
    
    st.markdown("""
    ### About the Model
    
    This application uses a **Machine Learning Classifier** trained on recipe data to predict traffic levels.
    The model is automatically loaded from the `model/` directory.
    
    #### Features Used:
    - **Calories**: Nutritional calorie content
    - **Carbohydrate**: Carbohydrate content in grams
    - **Sugar**: Sugar content in grams
    - **Protein**: Protein content in grams
    - **Category**: Recipe category (Target-encoded)
    - **Servings**: Number of servings
    
    #### Preprocessing Steps:
    1. Convert servings to numeric (handle text like "4 as a snack")
    2. Fill missing nutritional values with category-wise mean
    3. Target encode categorical features using trained encoder
    4. Standard scaling of all features
    
    """)
    
    if model and scaler and encoder:
        st.success("✅ All artifacts loaded successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Model Type:** {type(model).__name__}")
            st.info(f"**Number of Features:** 6")
            st.info(f"**Encoder Type:** {type(encoder).__name__}")
        
        with col2:
            st.info(f"**Scaling Method:** {type(scaler).__name__}")
            st.info("**Target Classes:** High Traffic (1), Low Traffic (0)")
            if CATEGORY_ENCODING:
                st.info(f"**Categories:** {len(CATEGORY_ENCODING)} supported")
        
        # Show metadata if available
        if metadata:
            st.markdown("---")
            st.markdown("### 📊 Model Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metadata.get('accuracy', 0):.4f}")
            with col2:
                st.metric("Precision", f"{metadata.get('precision', 0):.4f}")
            with col3:
                st.metric("Recall", f"{metadata.get('recall', 0):.4f}")
            with col4:
                st.metric("F1-Score", f"{metadata.get('f1_score', 0):.4f}")
        
        # Show supported categories
        if CATEGORY_ENCODING:
            st.markdown("---")
            st.markdown("### 📋 Supported Categories")
            
            # Display categories in a nice format
            cat_df = pd.DataFrame({
                'Category': list(CATEGORY_ENCODING.keys()),
                'Encoding Value': [f"{v:.4f}" for v in CATEGORY_ENCODING.values()]
            })
            st.dataframe(cat_df, use_container_width=True)
    else:
        st.error("❌ Model, scaler, or encoder not found. Please ensure the model files are in the model/ directory.")
    
    st.markdown("---")
    st.markdown("### 📝 Sample Data Format")
    
    sample_data = pd.DataFrame({
        'recipe': ['001', '002', '003'],
        'calories': [250.5, 450.0, 180.3],
        'carbohydrate': [35.2, 52.1, 28.5],
        'sugar': [12.5, 8.3, 15.2],
        'protein': [15.0, 22.5, 8.7],
        'category': ['Chicken', 'Dessert', 'Vegetable'],
        'servings': [4, 6, 2]
    })
    
    st.dataframe(sample_data, use_container_width=True)
    
    # Download sample CSV
    sample_csv = sample_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample CSV",
        data=sample_csv,
        file_name="sample_recipes.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🍳 Recipe Traffic Classifier | Built with Streamlit & CatBoost</p>
</div>
""", unsafe_allow_html=True)
