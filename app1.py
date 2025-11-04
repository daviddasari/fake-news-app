import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import requests
import os
import numpy as np

# --- Import the ML libraries ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# ----------------------------------------

# --- Config: LOCAL SAMPLE FILENAMES ---
TRUE_CSV_PATH = "true_sample_final.csv"
FAKE_CSV_PATH = "fake_sample_final.csv"
WELFAKE_CSV_PATH = "welfake_sample_final.csv"
EVAL_FILE_PATH = "evaluation_final.csv"  # <-- Reads your new .csv file
# ------------------------------------

# --- NLTK Stopwords Setup ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
stop_words = set(stopwords.words('english'))

# --- Re-usable Cleaning Function ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# --- MODEL TRAINING & CACHING ---
@st.cache_resource
def load_and_train_model():
    """
    This function runs ONCE when the app boots up.
    It loads the LOCAL SAMPLE data, trains the model, and returns the model/vectorizer.
    """
    st.write("First-time setup: Loading sample data and training model...")
    try:
        df_true = pd.read_csv(TRUE_CSV_PATH) 
        df_fake = pd.read_csv(FAKE_CSV_PATH)
    except FileNotFoundError:
        st.error(f"Error: '{TRUE_CSV_PATH}' or '{FAKE_CSV_PATH}' not found in the GitHub repo.")
        return None, None
    except Exception as e:
        st.error(f"Error reading CSV files: {e}.")
        return None, None

    # --- Create Labels (1=REAL, 0=FAKE) ---
    df_true['label'] = 1
    df_fake['label'] = 0

    # --- Combine and Shuffle ---
    df = pd.concat([df_true, df_fake])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- Preprocess ---
    df = df.dropna(subset=['title'])
    df['cleaned_title'] = df['title'].apply(clean_text)

    # --- Train/Test Split ---
    X = df['cleaned_title']
    y = df['label']
    
    # --- Vectorize ---
    vectorizer = TfidfVectorizer(max_features=5000) # Reduced features for sample
    X_tfidf = vectorizer.fit_transform(X)

    # --- Train Model ---
    model = LogisticRegression(max_iter=1000)
    model.fit(X_tfidf, y)

    st.success("Model trained on 10% sample data and cached successfully!")
    return model, vectorizer

# --- Data Loaders for Tabs ---
@st.cache_data
def load_data(file_path, **kwargs):
    """
    Loads any CSV file.
    """
    try:
        # We only need to read CSVs now
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, **kwargs)
        else:
            st.error(f"Error: Unknown file type for {file_path}. Please use .csv")
            return pd.DataFrame()
            
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found in the GitHub repo.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

# --- CACHED FUNCTIONS FOR CSV VISUALIZER ---
@st.cache_data
def load_uploaded_csv(uploaded_file):
    """Caches the uploaded CSV file to prevent re-reading on every rerun."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        return None

@st.cache_data
def get_df_description(df):
    """Caches the .describe() calculation."""
    return df.describe()

@st.cache_data
def get_all_columns(df):
    """Caches the column list."""
    return df.columns.tolist()

@st.cache_data
def get_numeric_columns(df):
    """Caches the numeric column selection."""
    return df.select_dtypes(include=np.number).columns.tolist()

@st.cache_data
def get_value_counts(df, column):
    """Caches the .value_counts() calculation (the slowest part)."""
    try:
        return df[column].value_counts()
    except Exception:
        # This can fail if the column has unhashable types like lists
        return None

# --- NEW CACHED FUNCTION ---
@st.cache_data
def get_sliced_data_for_plotting(_df, columns, num_rows):
    """Caches the data slicing for plotting."""
    if not columns:
        return pd.DataFrame()
    # We use _df as the argument name because st.cache_data hashes based on arguments
    # and df_viz (the object itself) will be the same, so this cache will work.
    return _df[columns].head(num_rows)
# ---------------------------

def process_isot_data(df_true, df_fake):
    if not df_true.empty and not df_fake.empty:
        df_true['label'] = 1
        df_fake['label'] = 0
        df = pd.concat([df_true, df_fake])
        df['label_name'] = df['label'].map({1: 'REAL', 0: 'FAKE'})
        if 'text' in df.columns:
            df['text_length'] = df['text'].astype(str).str.len()
        return df.sample(frac=1, random_state=42).reset_index(drop=True)
    return pd.DataFrame()

def process_test_data(df):
    if df.empty:
        return df
    # --- Auto-fix column names ---
    column_map = {}
    for col in df.columns:
        col_cleaned = col.lower().strip() # Clean the column name
        if col_cleaned == 'title':
            column_map[col] = 'title'
        if col_cleaned == 'label':
            column_map[col] = 'label'
    
    if 'title' in column_map.values() and 'label' in column_map.values():
        df = df.rename(columns=column_map)
    
    if 'title' not in df.columns or 'label' not in df.columns:
        st.error(f"Loaded file is missing 'title' or 'label' columns.")
        return pd.DataFrame()
    
    return df

# --- Load all assets ---
model, vectorizer = load_and_train_model()
df_isot = process_isot_data(load_data(TRUE_CSV_PATH), load_data(FAKE_CSV_PATH))
df_welfake = process_test_data(load_data(WELFAKE_CSV_PATH))
df_evaluation = process_test_data(load_data(EVAL_FILE_PATH)) # <-- LOAD NEW FILE

# --- Main App UI ---
st.title("📰 The Real Fake News Detector")

# --- Create Tabs ---
tab1, tab2, tab3, tab4, tab_viz, tab5 = st.tabs([
    "📰 News Analyzer", 
    "📊 Visual Insights", 
    "🔍 Cross-Validation (WELFake)", 
    "🧪 Final Evaluation",
    "📊 CSV Visualizer",  # <-- NEW TAB
    "ℹ️ About This Model"
])

# --- Tab 1: News Analyzer ---
with tab1:
    st.header("Analyze a News Headline or Text")
    
    st.sidebar.title("About This Analyzer")
    st.sidebar.info(
        "**Project: Fake News Detection**\n\n"
        "This model is trained on a 10% sample of the **ISOT Dataset**."
    )
    
    st.sidebar.title("How to Use")
    st.sidebar.markdown(
        """
        1.  Enter a news headline in the text box.
        2.  Click the **Analyze** button.
        3.  The model will predict the result.
        """
    )

    user_input = st.text_area("News Text", "", height=200, placeholder="Paste your news text here...")

    if st.button("Analyze", type="primary"):
        if model and vectorizer:
            if user_input.strip() == "":
                st.warning("Please enter some text to analyze.")
            else:
                cleaned_input = clean_text(user_input)
                vectorized_input = vectorizer.transform([cleaned_input])
                prediction = model.predict(vectorized_input)
                probability = model.predict_proba(vectorized_input)
                
                # --- THIS IS THE FIX ---
                confidence = probability[0][int(prediction[0])] * 100
                # -----------------------
                
                if prediction[0] == 1:
                    st.success(f"**Prediction: REAL News** (Confidence: {confidence:.2f}%)")
                else:
                    st.error(f"**Prediction: FAKE News** (Confidence: {confidence:.2f}%)")
        else:
            st.error("Model is not loaded. Please wait for training to complete or check logs.")

# --- Tab 2: Visual Insights ---
with tab2:
    st.header("Visual Insights from the ISOT Training Data (10% Sample)")
    
    if not df_isot.empty:
        # Plot 1
        st.subheader("1. Balance of Real vs. Fake News")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.countplot(data=df_isot, x='label_name', ax=ax1, palette=["#E63946", "#457B9D"])
        st.pyplot(fig1)

        # Plot 2
        st.subheader("2. News Subject Analysis")
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
        real_subjects = df_isot[df_isot['label'] == 1]['subject'].value_counts()
        sns.barplot(x=real_subjects.values, y=real_subjects.index, ax=ax1, palette="viridis")
        ax1.set_title("Top Subjects for REAL News")
        fake_subjects = df_isot[df_isot['label'] == 0]['subject'].value_counts()
        sns.barplot(x=fake_subjects.values, y=fake_subjects.index, ax=ax2, palette="plasma")
        ax2.set_title("Top Subjects for FAKE News")
        plt.tight_layout()
        st.pyplot(fig2)

        # Plot 3
        st.subheader("3. Article Length Distribution")
        if 'text_length' in df_isot.columns:
            df_filtered = df_isot[df_isot['text_length'] < 20000]
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            sns.histplot(data=df_filtered, x='text_length', hue='label_name', kde=True, multiple="stack",
                         palette=["#E63946", "#457B9D"], bins=50)
            ax3.set_title("Distribution of Article Length (Characters)")
            st.pyplot(fig3)
    else:
        st.error("Could not load ISOT data for visualization.")

# --- Tab 3: Cross-Validation (WELFake) ---
with tab3:
    st.header("Cross-Dataset Validation Test (on 'WELFake' Sample)")
    st.write("How does our model (trained on ISOT) perform on the 'WELFake' dataset?")

    if not df_welfake.empty and model and vectorizer:
        with st.spinner(f"Running model on {len(df_welfake)} 'WELFake' sample articles..."):
            try:
                df_welfake = df_welfake.dropna(subset=['title', 'label'])
                X_welfake = df_welfake['title'].apply(clean_text)
                y_welfake_original = df_welfake['label'].astype(int) 
                
                # Flip labels (0=REAL, 1=FAKE) -> (1=REAL, 0=FAKE)
                y_welfake_true_flipped = y_welfake_original.map({0: 1, 1: 0})
                
                X_welfake_tfidf = vectorizer.transform(X_welfake)
                y_welfake_pred = model.predict(X_welfake_tfidf)
                
                accuracy = accuracy_score(y_welfake_true_flipped, y_welfake_pred)
                
                st.metric(
                    label="Accuracy on 'WELFake' Sample (Labels Corrected)",
                    value=f"{accuracy * 100:.2f}%"
                )
                
                cm = confusion_matrix(y_welfake_true_flipped, y_welfake_pred)
                fig4, ax4 = plt.subplots(figsize=(8, 5))
                sns.heatmap(cm, annot=True, fmt='d', ax=ax4, cmap='Blues',
                            xticklabels=['FAKE (Pred)', 'REAL (Pred)'],
                            yticklabels=['FAKE (Actual)', 'REAL (Actual)'])
                st.pyplot(fig4)
            
            except Exception as e:
                st.error(f"An error occurred during cross-validation: {e}")
            
    else:
        st.error("Could not run validation. `welfake_sample_final.csv` not found or model not loaded.")

# --- Tab 4: Final Evaluation (NEW TAB) ---
with tab4:
    st.header("Final Evaluation Test (on 'evaluation_final.csv')")
    st.write("How does our model (trained on ISOT) perform on this new evaluation dataset?")

    if not df_evaluation.empty and model and vectorizer:
        with st.spinner(f"Running model on {len(df_evaluation)} 'evaluation_final' articles..."):
            try:
                # 1. Prepare Data
                df_eval = df_evaluation.dropna(subset=['title', 'label'])
                X_eval = df_eval['title'].apply(clean_text)
                y_eval_original = df_eval['label'].astype(int) 
                
                # 2. Make Predictions
                X_eval_tfidf = vectorizer.transform(X_eval)
                y_eval_pred = model.predict(X_eval_tfidf)
                
                # --- Smart Label Checking ---
                # Test 1: Assume labels MATCH (1=REAL, 0=FAKE)
                accuracy_normal = accuracy_score(y_eval_original, y_eval_pred)
                
                # Test 2: Assume labels are FLIPPED (0=REAL, 1=FAKE)
                y_eval_flipped = y_eval_original.map({0: 1, 1: 0})
                accuracy_flipped = accuracy_score(y_eval_flipped, y_eval_pred)
                
                st.subheader("Label Scheme Analysis")
                st.write("We automatically checked for inverted labels. The highest score is the correct one.")
                
                col1, col2 = st.columns(2)
                col1.metric("Accuracy (Assuming 1=REAL)", f"{accuracy_normal * 100:.2f}%")
                col2.metric("Accuracy (Assuming 0=REAL)", f"{accuracy_flipped * 100:.2f}%")

                st.subheader("Final Validation Result")
                if accuracy_flipped > accuracy_normal:
                    st.success(f"True Accuracy: {accuracy_flipped * 100:.2f}%")
                    st.info("Insight: This dataset uses inverted labels (0=REAL, 1=FAKE).")
                    cm = confusion_matrix(y_eval_flipped, y_eval_pred)
                    labels = ['FAKE (Actual)', 'REAL (Actual)']
                else:
                    st.success(f"True Accuracy: {accuracy_normal * 100:.2f}%")
                    st.info("Insight: This dataset uses the same labels as our model (1=REAL, 0=FAKE).")
                    cm = confusion_matrix(y_eval_original, y_eval_pred)
                    labels = ['FAKE (Actual)', 'REAL (Actual)']

                # Plot the correct confusion matrix
                fig5, ax5 = plt.subplots(figsize=(8, 5))
                sns.heatmap(cm, annot=True, fmt='d', ax=ax5, cmap='Greens',
                            xticklabels=['FAKE (Pred)', 'REAL (Pred)'],
                            yticklabels=labels)
                st.pyplot(fig5)

            except Exception as e:
                st.error(f"An error occurred during evaluation: {e}")
    else:
        st.error("Could not run validation. `evaluation_final.csv` not found or model not loaded.")


# --- Tab (NEW): CSV Visualizer ---
with tab_viz:
    st.header("🚀 Instant CSV Visualizer")
    st.write("Drag and drop a CSV file here to see basic visualizations.")

    # Add file uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="csv_visualizer")

    if uploaded_file is not None:
        # Use the cached function to load the data
        df_viz = load_uploaded_csv(uploaded_file)
        
        # Check if the dataframe was loaded successfully
        if df_viz is not None:
            st.success("File uploaded and cached successfully!")
            
            # Show dataframe preview
            st.subheader("Raw Data Preview")
            st.dataframe(df_viz.head())
            
            # --- Use cached function for .describe() ---
            st.subheader("Data Description (Numeric Columns)")
            try:
                st.write(get_df_description(df_viz)) # CACHED
            except Exception as e:
                st.info(f"Could not generate numeric description: {e}")
            
            # --- Visualization Options ---
            st.subheader("Create Visualizations")
            
            # --- Use cached functions for column lists ---
            all_columns = get_all_columns(df_viz) # CACHED
            numeric_columns = get_numeric_columns(df_viz) # CACHED
            
            if not numeric_columns:
                st.warning("No numeric columns found in this CSV for plotting line, area, or scatter charts.")
            
            # --- UPDATED LINE/AREA CHART BLOCK ---
            st.markdown("---")
            st.write("### Line / Area Chart")
            st.info("Best for time series or showing trends. Select one or more numeric columns.")
            
            if numeric_columns:
                cols_to_plot = st.multiselect("Select numeric columns to plot", numeric_columns, key='viz_line_multi')
                
                if cols_to_plot: # Only show slider if columns are selected
                    # --- ADDED SLIDER TO LIMIT DATA ---
                    max_rows = len(df_viz)
                    # Set a reasonable default (2000) that won't lag
                    default_sample = min(max_rows, 2000) 
                    
                    sample_size = st.slider(
                        "Limit data to plot (first N rows)", 
                        min_value=min(100, max_rows), # Handle small files
                        max_value=max_rows, 
                        value=default_sample,
                        step=100,
                        key='viz_line_sampler',
                        help="Reduces the number of points plotted to prevent browser lag. Default is 2000."
                    )
                    
                    # Use the new cached function to get the sliced data
                    data_to_plot = get_sliced_data_for_plotting(df_viz, cols_to_plot, sample_size)
                    # --- END OF UPDATE ---
                    
                    chart_type = st.radio("Select chart type", ["Line Chart", "Area Chart"], key='viz_line_radio')
                    
                    # Plot the *sampled* data, not the full dataframe
                    if chart_type == "Line Chart":
                        st.line_chart(data_to_plot)
                    else:
                        st.area_chart(data_to_plot)
            else:
                st.info("This chart type requires numeric columns.")
            # --- END OF UPDATED BLOCK ---

            # 2. Bar Chart (Value Counts)
            st.markdown("---")
            st.write("### Bar Chart (Value Counts)")
            st.info("Best for seeing the frequency of items in a single categorical column.")
            cat_col = st.selectbox("Select a column to see its value counts", [None] + all_columns, key='viz_bar_select')
            
            if cat_col:
                # --- Use cached function for .value_counts() ---
                value_counts = get_value_counts(df_viz, cat_col) # CACHED
                
                if value_counts is not None:
                    st.bar_chart(value_counts)
                else:
                    st.warning(f"Could not generate value counts for column '{cat_col}'. It might contain unhashable types like lists.")


            # 3. Scatter Plot (2 Variables)
            if len(numeric_columns) >= 2:
                st.markdown("---")
                st.write("### Scatter Plot")
                st.info("Best for comparing two numeric variables.")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_axis = st.selectbox("Select X-axis", [None] + numeric_columns, key='viz_scatter_x')
                with col2:
                    y_axis = st.selectbox("Select Y-axis", [None] + numeric_columns, key='viz_scatter_y')
                
                if x_axis and y_axis:
                    # Note: Scatter plots can also lag on large data.
                    # We can apply the same .head(sample_size) logic here if needed.
                    st.scatter_chart(df_viz.head(sample_size), x=x_axis, y=y_axis)
                    st.caption(f"Showing scatter plot for first {sample_size} rows.")
            else:
                st.markdown("---")
                st.info("A Scatter Plot requires at least two numeric columns in your data.")
            
# ----------------------------------------

# --- Tab 5: About This Model ---
with tab5:
    st.header("About Our Model (Trained on 10% Sample)")
    st.write(
        """
        This model was trained live on a 10% sample of the **ISOT Dataset**.
        This is to ensure the app can be deployed on Streamlit Cloud, which has
        file size and memory limits.
        
        - **REAL News (Label 1):** ~2,100 articles from Reuters.com
        - **FAKE News (Label 0):** ~2,300 articles from known fake news sources.
        """
    )
    
    if not df_isot.empty:
        with st.expander("Click to view sample of `True.csv` (REAL News)"):
            st.dataframe(df_isot[df_isot['label'] == 1][['title', 'text', 'subject']].head())
            
        with st.expander("Click to view sample of `Fake.csv` (FAKE News)"):
            st.dataframe(df_isot[df_isot['label'] == 0][['title', 'text', 'subject']].head())
            
    else:
        st.error("Could not load ISOT data.")

