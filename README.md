📰 Fake News Detection Dashboard

This is an interactive web application for a Data Analytics and Visualization project. The app analyzes news headlines to detect "Real" vs. "Fake" news using a Logistic Regression model.

This project demonstrates a complete machine learning workflow, from baseline modeling and data analysis to model refinement, cross-validation, and final deployment.

Project Analysis & Model Refinement

The project was executed in several deliberate phases to build a robust and reliable model.

Baseline Model: An initial model was trained on a preliminary dataset (merged_news.csv) to establish feasibility and a performance baseline.

Data Analysis & Refinement: Analysis of the baseline model's predictions revealed significant inconsistencies in the original dataset's labels. For example, many verified headlines from sources like (Reuters) were incorrectly labeled as "FAKE".

The Solution (ISOT Dataset): To build a reliable classifier, the model was retrained on the high-quality, industry-standard ISOT Dataset. This dataset correctly identifies trusted sources, resolving the core bias of the initial data.

The Deployment Model: To balance performance with cloud deployment constraints (GitHub file size limits), this final app is trained on a 10% representative sample of the ISOT dataset. The model trains live when the app first boots up using Streamlit's @st.cache_resource.

Features

The app is organized into five tabs, reflecting a full data analysis workflow:

1. 📰 News Analyzer

A simple interface to paste in any news headline. The model will predict whether it's "REAL" or "FAKE" and provide a confidence score.

2. 📊 Visual Insights

Analyzes the ISOT (10% sample) training data. This tab shows:

The class balance of "REAL" vs. "FAKE" news in the training set.

A breakdown of article subjects (e.g., politicsNews, worldnews), showing the clear distinction our model learned.

The distribution of article lengths.

3. 🔍 Cross-Validation (WELFake)

Validates our model (trained on ISOT) against the welfake_sample_final.csv dataset. This tab:

Automatically identifies and corrects for the WELFake dataset's inverted label scheme (0=REAL, 1=FAKE).

Calculates the model's "true" accuracy against this new data.

Displays a confusion matrix of the validation results.

4. 🧪 Final Evaluation

Performs a final accuracy test against a third dataset, evaluation_final.csv. This tab:

Automatically checks the label scheme (flipped or normal) to ensure a correct comparison.

Calculates the final validation accuracy.

Displays the final confusion matrix.

5. ℹ️ About This Model

Explains the data, the 10% sampling for deployment, and the model's architecture.

How to Run Locally

Clone the repository:

git clone [https://github.com/daviddasari/fake-news-app.git](https://github.com/daviddasari/fake-news-app.git)
cd fake-news-app


Install the required libraries:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app1.py
