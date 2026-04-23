import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ==========================================
# PAGE CONFIGURATION (Must be first)
# ==========================================
st.set_page_config(page_title="Student Performance Analysis", page_icon="🎓", layout="wide")

# ADD THIS TO LOAD THE BANNER IMAGE
try:
    st.image("banner.png", use_container_width=True)
except FileNotFoundError:
    pass # If the image isn't found, it just silently skips it

# ==========================================
# LOAD DATA AND MODELS (Cached for speed)
# ==========================================
@st.cache_resource # This prevents the model from reloading on every click
def load_models():
    try:
        model = joblib.load("model/student_model.pkl")
        scaler = joblib.load("model/scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run your training script first.")
        return None, None

@st.cache_data
def load_data():
    try:
        # 1. Load the data
        df = pd.read_csv("data/student_performance_200.csv")
        
        # 2. Unbreakable result_clean creation
        if 'result' in df.columns:
            # Check if the column contains words like "Pass" or "Fail"
            first_value = str(df['result'].iloc[0]).strip().lower()
            
            if first_value in ['pass', 'fail']:
                # Convert words to 1 and 0 mathematically
                df['result_clean'] = df['result'].astype(str).str.strip().str.lower().map({'fail': 0, 'pass': 1})
            else:
                # If they are already numbers, just force them to be numeric
                df['result_clean'] = pd.to_numeric(df['result'], errors='coerce')
                
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset not found in the 'data/' folder!")
        return pd.DataFrame()

model, scaler = load_models()
df = load_data()


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🎓 Navigation Menu")
st.sidebar.markdown("Select a module below:")

# Creating the navigation buttons
page = st.sidebar.radio(
    "Go to:",
    ["📊 Basic Analysis (EDA)", "🧠 Model Explainability", "🔮 Predict Performance", "🚨 At-Risk Dashboard"]
)

st.sidebar.divider()
st.sidebar.info("Built by - [Shivek Singh]")

# ==========================================
# PAGE 1: ADVANCED ANALYSIS (EDA)
# ==========================================
if page == "📊 Basic Analysis (EDA)":
    st.title("📊 Advanced Data Exploration")
    st.write("Deep dive into student metrics, distributions, and hidden patterns.")

    # Set Seaborn theme for highly attractive plots
    sns.set_theme(style="whitegrid", palette="muted")

    if not df.empty:
        # --- SECTION 1: DATA HEALTH & SUMMARY ---
        st.markdown("### 🛠️ 1. Data Health & Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Students", len(df))
        col2.metric("Total Features", len(df.columns) - 1) # Excluding target
        col3.metric("Missing Values", df.isnull().sum().sum(), delta_color="inverse")
        col4.metric("Pass Rate", f"{(df['result_clean'].mean() * 100):.1f}%")

        with st.expander("📌 View Detailed Descriptive Statistics"):
            st.dataframe(df.describe().T.style.background_gradient(cmap='Blues'), use_container_width=True)

        st.divider()

        # --- SECTION 2: TARGET & CATEGORICAL DISTRIBUTIONS ---
        st.markdown("### 🎯 2. Cohort Demographics & Target Balance")
        
        # Dynamically find categorical columns (like Gender, Ethnicity, if they exist)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.drop('result', errors='ignore').tolist()
        
        col_pie, col_cat = st.columns([1, 2])
        
        with col_pie:
            st.write("**Overall Pass vs Fail**")
            fig, ax = plt.subplots(figsize=(5, 5))
            # Attractive Donut Chart
            result_counts = df['result'].value_counts()
            ax.pie(result_counts, labels=result_counts.index, autopct='%1.1f%%', 
                   startangle=90, colors=['#ff9999','#66b3ff'], wedgeprops=dict(width=0.4, edgecolor='w'))
            st.pyplot(fig)

        with col_cat:
            if cat_cols:
                st.write("**Categorical Breakdown**")
                selected_cat = st.selectbox("Select Demographic to Analyze:", cat_cols)
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.countplot(y=selected_cat, hue='result', data=df, palette='Set2', ax=ax)
                sns.despine()
                st.pyplot(fig)
            else:
                st.info("💡 No extra demographic data (e.g., Gender, Race) found in this dataset. If you upload a richer CSV, demographic bar charts will automatically appear here!")

        st.divider()

        # --- SECTION 3: ADVANCED FEATURE DISTRIBUTIONS ---
        st.markdown("### 📈 3. Feature Distributions & Performance Spreads")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.drop('result_clean', errors='ignore').tolist()
        
        col_dist1, col_dist2 = st.columns([1, 2])
        with col_dist1:
            st.write("**Univariate Distribution**")
            feature_choice = st.radio("Select a numeric feature:", numeric_cols)
            
        with col_dist2:
            fig, ax = plt.subplots(figsize=(10, 4))
            # KDE + Histogram combination is much more professional
            sns.histplot(data=df, x=feature_choice, hue='result', kde=True, 
                         palette='magma', element='step', stat='density', common_norm=False, ax=ax)
            ax.set_title(f"Distribution of {feature_choice.replace('_', ' ').title()} by Result")
            sns.despine()
            st.pyplot(fig)

        st.divider()

        # --- SECTION 4: RELATIONSHIPS & CORRELATION ---
        st.markdown("### 🔗 4. Advanced Relationships")
        
        col_rel1, col_rel2 = st.columns(2)
        
        with col_rel1:
            st.write("**Box & Scatter Plot (Summary + Points)**")
            st.caption("Shows the statistical average (box) alongside every individual student (dots).")
            fig, ax = plt.subplots(figsize=(6, 5))
            
            # 1. The Boxplot (semi-transparent so it doesn't hide the dots)
            sns.boxplot(
                x='result', 
                y=feature_choice, 
                data=df, 
                palette='pastel', 
                boxprops={'alpha': 0.6}, # Makes the box see-through
                showfliers=False,        # Hide default boxplot outliers to avoid double-printing dots
                ax=ax
            )
            
            # 2. The Strip Plot (plots the actual students as dots with a slight 'jitter' so they don't overlap)
            sns.stripplot(
                x='result', 
                y=feature_choice, 
                data=df, 
                color='#2c3e50',         # Dark professional grey/blue for the dots
                alpha=0.7,               # Slight transparency on the dots
                jitter=True,             # Spreads the dots out horizontally
                size=5, 
                ax=ax
            )
            
            sns.despine()
            st.pyplot(fig)

        with col_rel2:
            st.write("**Feature Correlation Matrix**")
            st.caption("Identifies strong mathematical relationships between study metrics.")
            fig, ax = plt.subplots(figsize=(6, 5))
            # Create a mask to hide the upper triangle (looks cleaner)
            corr = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", 
                        linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
            st.pyplot(fig)

        # Jointplot (Interactive-style scatter)
        st.write("**Bivariate Scatter Analysis**")
        if len(numeric_cols) >= 2:
            col_x, col_y = st.columns(2)
            with col_x:
                x_axis = st.selectbox("X-Axis Feature:", numeric_cols, index=0)
            with col_y:
                y_axis = st.selectbox("Y-Axis Feature:", numeric_cols, index=1 if len(numeric_cols)>1 else 0)
            
            fig = sns.jointplot(data=df, x=x_axis, y=y_axis, hue='result', palette='husl', alpha=0.7, height=6)
            st.pyplot(fig)


# ==========================================
# PAGE 2: MODEL EXPLAINABILITY (New!)
# ==========================================
elif page == "🧠 Model Explainability":
    st.title("🧠 How the AI Thinks")
    st.write("Machine Learning shouldn't be a black box. Here is exactly how the model makes its decisions.")

    if model and scaler:
        st.subheader("Feature Importance Weights")
        st.write("This chart shows the exact mathematical weight the Logistic Regression model assigns to each metric. Larger bars mean that feature has a bigger impact on passing.")

        # Extracting coefficients from the trained model
        coefficients = model.coef_[0]
        feature_names = ['Study Hours', 'Attendance', 'Previous Score', 'Assignments Completed']
        
        # Create a DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Weight': coefficients
        })
        # Sort by weight for a cleaner chart
        importance_df = importance_df.sort_values(by='Weight', ascending=True)

        # Plotting the Feature Importance
        fig, ax = plt.subplots(figsize=(8, 4))
        # Use a color palette based on values (positive=greenish/blue)
        sns.barplot(x='Weight', y='Feature', data=importance_df, palette='mako', ax=ax)
        ax.set_xlabel("Impact on Passing (Logistic Weight)")
        ax.set_ylabel("")
        st.pyplot(fig)

        st.divider()
        
        # Adding a professional Data Science insight box
        st.info("### 👨‍💻 Data Scientist Insight:")
        
        # Find the most important feature dynamically
        top_feature = importance_df.iloc[-1]['Feature']
        st.write(f"Based on the algorithm's weights, **{top_feature}** is the strongest predictor of student success in this dataset. When consulting with educators, we should recommend policies that primarily target improving {top_feature.lower()} before focusing on other metrics.")


# ==========================================
# PAGE 3: PREDICT PERFORMANCE
# ==========================================
elif page == "🔮 Predict Performance":
    st.title("🔮 Predict Student Performance")
    st.write("Enter the student's current metrics to predict their final result.")

    if model and scaler:
        # Create a clean form layout
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                study_hours = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
                attendance = st.number_input("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
            with col2:
                previous_score = st.number_input("Previous Score (/100)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
                assignments = st.number_input("Assignments Completed", min_value=0, max_value=50, value=5, step=1)
            
            submit_button = st.form_submit_button("Run Prediction Engine")

        if submit_button:
            user_data = pd.DataFrame(
                [[study_hours, attendance, previous_score, assignments]], 
                columns=['study_hours', 'attendance', 'previous_score', 'assignments_completed']
            )
            
            scaled_data = scaler.transform(user_data)
            prediction = model.predict(scaled_data)[0]
            probability = model.predict_proba(scaled_data)[0][1] * 100

            st.divider()
            if prediction == 1:
                # Using Deep Violet (#6D28D9) for success instead of green
                st.markdown(f"""
                <div style='background-color: #F3E8FF; padding: 20px; border-left: 5px solid #6D28D9; border-radius: 5px;'>
                    <h3 style='color: #6D28D9; margin:0;'>🎓 Prediction: PASS</h3>
                    <p style='color: #1F2937; margin-top: 5px;'><b>Confidence:</b> {probability:.1f}% chance of passing.</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                # Using Soft Coral (#F472B6) for alert instead of harsh red
                st.markdown(f"""
                <div style='background-color: #FDF2F8; padding: 20px; border-left: 5px solid #F472B6; border-radius: 5px;'>
                    <h3 style='color: #BE185D; margin:0;'>⚠️ Prediction: AT-RISK (FAIL)</h3>
                    <p style='color: #1F2937; margin-top: 5px;'><b>Current Chance of Passing:</b> {probability:.1f}%</p>
                    <p style='color: #666666; font-size: 14px;'><i>Go to the 'At-Risk Dashboard' tab to see intervention strategies.</i></p>
                </div>
                """, unsafe_allow_html=True)



# ==========================================
# PAGE 4: AT-RISK DASHBOARD (New!)
# ==========================================
elif page == "🚨 At-Risk Dashboard":
    st.title("🚨 Class Intervention Dashboard")
    st.write("Automatically identify students who are failing but are **extremely close to passing**. These are the students where teacher intervention will have the highest ROI.")

    if model and scaler and not df.empty:
        # Extract the features for all students in the CSV
        features_df = df[['study_hours', 'attendance', 'previous_score', 'assignments_completed']]
        
        # Predict the exact passing probability for EVERY student
        scaled_features = scaler.transform(features_df)
        probabilities = model.predict_proba(scaled_features)[:, 1] * 100
        
        # Create a new analytical dataframe
        risk_df = df.copy()
        risk_df['Pass Probability (%)'] = np.round(probabilities, 1)
        
        # Filter 1: Find students who are predicted to FAIL (Probability < 50%)
        failing_students = risk_df[risk_df['Pass Probability (%)'] < 50.0]
        
        # Filter 2: Sort them by probability descending (so the closest to 50% are at the top)
        borderline_students = failing_students.sort_values(by='Pass Probability (%)', ascending=False)
        
        # UI Metrics
        total_students = len(df)
        failing_count = len(failing_students)
        borderline_count = len(borderline_students[borderline_students['Pass Probability (%)'] > 35.0]) # Arbitrary 35% threshold for "close"
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Students", total_students)
        col2.metric("Total Failing", failing_count)
        col3.metric("Borderline (High Priority)", borderline_count, delta="Requires Intervention", delta_color="inverse")

        st.divider()
        st.subheader("Priority Intervention List")
        st.write("These students are failing, but have the highest chance of passing if given extra help.")
        
        # Format the dataframe to look beautiful in Streamlit
        st.dataframe(
            borderline_students[['study_hours', 'attendance', 'previous_score', 'assignments_completed', 'Pass Probability (%)']].head(15),
            use_container_width=True,
            hide_index=True
        )

        # Actionable insight for the teacher
        st.warning(f"🔔 **Action Item:** The top student in this list has a **{borderline_students.iloc[0]['Pass Probability (%)']}%** chance of passing. A minor increase in their study hours or completing one missing assignment could push them over the 50% threshold.")