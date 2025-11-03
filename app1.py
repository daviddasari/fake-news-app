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

# --- Import the ML libraries ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# ----------------------------------------

# --- Config: LOCAL SAMPLE FILENAMES ---
# These files MUST be in your GitHub repo.
TRUE_CSV_PATH = "true_sample_final.csv"
FAKE_CSV_PATH = "fake_sample_final.csv"
WELFAKE_CSV_PATH = "welfake_sample_final.csv"
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
    
    # --- Load Data ---
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

# --- DATA LOADING FOR VISUALIZATION ---
@st.cache_data
def load_visualization_data():
    """
    Loads and combines the CSV files for visualization.
    """
    try:
        df_true = pd.read_csv(TRUE_CSV_PATH)
        df_fake = pd.read_csv(FAKE_CSV_PATH)
        
        df_true['label'] = 1
        df_fake['label'] = 0
        df = pd.concat([df_true, df_fake])
        
        df['label_name'] = df['label'].map({1: 'REAL', 0: 'FAKE'})
        if 'text' in df.columns:
            df['text_length'] = df['text'].astype(str).str.len()
        
        return df.sample(frac=1, random_state=42).reset_index(drop=True)
    except FileNotFoundError:
        st.error(f"Error: '{TRUE_CSV_PATH}' or '{FAKE_CSV_PATH}' not found.")
        return pd.DataFrame()

# --- DATA LOADING FOR CROSS-VALIDATION ---
@st.cache_data
def load_welfake_data():
    """
    Loads the welfake_sample_final.csv file for cross-validation.
    """
    try:
        df = pd.read_csv(WELFAKE_CSV_PATH)
        
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
            st.error(f"`{WELFAKE_CSV_PATH}` is missing 'title' or 'label' columns.")
            return pd.DataFrame()
            
        return df
    except FileNotFoundError:
        st.error(f"Error: '{WELFAKE_CSV_PATH}' not found.")
        return pd.DataFrame()

# --- Load all assets ---
model, vectorizer = load_and_train_model()
df_isot = load_visualization_data()
df_welfake = load_welfake_data()

# --- Main App UI ---
st.title("📰 The Real Fake News Detector")

# --- Create Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📰 News Analyzer", "📊 Visual Insights", "🔍 Cross-Validation", "ℹ️ About This Model"])

# --- Tab 1: News Analyzer ---
with tab1:
    st.header("Analyze a News Headline or Text")
    
    st.sidebar.title("About This Analyzer")
    st.sidebar.info(
        "**Project: Fake News Detection**\n\n"
        "This model is trained on a 10% sample of the **ISOT Dataset** "
        "(which includes Reuters as 'REAL' news)."
    )
    
    st.sidebar.title("How to Use")
    st.sidebar.markdown(
        """
        1.  Enter a news headline in the text box.
        2.  Click the **Analyze** button.
        3.  The model will predict the result.
        """
    )

    user_input = st.text_area(
        "News Text", 
        "", 
        height=200, 
        placeholder="Paste your news text here..."
    )

    if st.button("Analyze", type="primary"):
        if model and vectorizer:
            if user_input.strip() == "":
                st.warning("Please enter some text to analyze.")
            else:
                cleaned_input = clean_text(user_input)
                vectorized_input = vectorizer.transform([cleaned_input])
                prediction = model.predict(vectorized_input)
                probability = model.predict_proba(vectorized_input)
                confidence = probability[0][prediction[0]] * 100
                
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
        st.subheader("1. Balance of Real vs. Fake News")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.countplot(data=df_isot, x='label_name', ax=ax1, palette=["#E63946", "#457B9D"])
        st.pyplot(fig1)

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

# --- Tab 3: Cross-Validation ---
with tab3:
    st.header("Cross-Dataset Validation Test (on 10% Sample)")
    st.write("How does our model (trained on ISOT) perform on the 'WELFake' dataset?")

    if not df_welfake.empty and model and vectorizer:
        with st.spinner(f"Running model on {len(df_welfake)} 'WELFake' sample articles..."):
            try:
                df_welfake = df_welfake.dropna(subset=['title', 'label'])
                X_welfake = df_welfake['title'].apply(clean_text)
                y_welfake_original = df_welfake['label'].astype(int) 
                
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

# --- Tab 4: About This Model ---
with tab4:
    st.header("About Our Model (Trained on 10% Sample)")
    st.write(
        """
        This model was trained live on a 10% sample of the **ISOT Dataset**.
        This is to ensure the app can be deployed on Streamlit Cloud, which has
        file size and memory limits.
        
        - **REAL News (Label 1):** ~2,100 articles from Reuters.com
        - **FAKE News (Label 0):** ~2,300 articles from known fake news sources.
        
        The data is loaded locally from the GitHub repo.
        """
    )
    
    if not df_isot.empty:
        with st.expander("Click to view sample of `True.csv` (REAL News)"):
            st.dataframe(df_isot[df_isot['label'] == 1][['title', 'text', 'subject']].head())
            
        with st.expander("Click to view sample of `Fake.csv` (FAKE News)"):
            st.dataframe(df_isot[df_isot['label'] == 0][['title', 'text', 'subject']].head())
            
    else:
        st.error("Could not load ISOT data.")

