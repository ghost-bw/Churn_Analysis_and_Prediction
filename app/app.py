import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------- PAGE SETUP -----------------
st.set_page_config(
    page_title="ChurnLens | Customer Churn Analytics",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM STYLE -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    h1, h2, h3, h4, h5, h6 { font-family: 'Outfit', sans-serif; }

    .brand-title {
        font-family: 'Outfit', sans-serif; font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8F8F 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem; text-align: center;
    }
    .brand-subtitle {
        font-size: 0.9rem; color: var(--text-color); opacity: 0.6;
        text-align: center; margin-bottom: 1.5rem; font-style: italic;
    }
    .glass-card {
        border-radius: 12px; padding: 24px;
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover { transform: translateY(-2px); border-color: rgba(255, 75, 75, 0.4); }
    .custom-metric {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px; padding: 16px; text-align: center;
        transition: all 0.3s ease; box-shadow: 0 2px 10px rgba(0,0,0,0.02);
    }
    .custom-metric:hover { background: rgba(128, 128, 128, 0.05); transform: translateY(-1px); }
    .custom-metric-title {
        font-size: 0.85rem; color: var(--text-color); opacity: 0.7;
        text-transform: uppercase; font-weight: 600; letter-spacing: 0.8px; margin-bottom: 6px;
    }
    .custom-metric-value { font-size: 1.8rem; font-weight: 700; color: var(--text-color); }
    .custom-metric-accent { color: #FF4B4B !important; }
    .risk-badge {
        font-size: 1.1rem; font-weight: 700; padding: 6px 12px;
        border-radius: 20px; display: inline-block; margin-bottom: 15px;
    }
    .badge-low  { background-color: rgba(46,204,113,0.15); border: 1px solid #2ecc71; color: #27ae60; }
    .badge-medium { background-color: rgba(241,196,15,0.15); border: 1px solid #f1c40f; color: #d4ac0d; }
    .badge-high { background-color: rgba(231,76,60,0.15); border: 1px solid #e74c3c; color: #c0392b; }
    .recommendation-box {
        background: rgba(128,128,128,0.03); border: 1px dashed rgba(128,128,128,0.2);
        border-radius: 8px; padding: 15px; margin-top: 10px;
    }
    .info-item { margin-bottom: 8px; font-size: 0.95rem; }
    .info-label { font-weight: 600; color: var(--text-color); opacity: 0.7; }
    .info-value { color: var(--text-color); }
    .section-spacing { margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ----------------- DATA & MODEL LOADERS -----------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for fname in ['logistic_regression_best_model.joblib', 'logistic_regression_best_model.pkl']:
        path = os.path.join(base_dir, 'models', fname)
        if os.path.exists(path):
            try:
                m = joblib.load(path)
                for _, transformer, _ in m.named_steps['preprocessor'].transformers_:
                    if hasattr(transformer, 'named_steps') and 'imputer' in transformer.named_steps:
                        imp = transformer.named_steps['imputer']
                        if hasattr(imp, '_fit_dtype') and not hasattr(imp, '_fill_dtype'):
                            imp._fill_dtype = imp._fit_dtype
                return m
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None
    st.error("Model file not found in models/")
    return None


@st.cache_data
def load_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for fname in ['cleaned_churn_data.csv', 'exported_churn_data.csv']:
        path = os.path.join(base_dir, 'data', fname)
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception as e:
                st.error(f"Error reading dataset: {e}")
                return None
    st.error("Dataset file not found in data/")
    return None


@st.cache_data
def load_feature_importance():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, 'reports', 'feature_importance.csv')
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


@st.cache_data
def get_sample_csv_template():
    sample = pd.DataFrame([
        {'customerid':'CUST-0091','customer_name':'Karan Malhotra','gender':'Male','dob':'1987-04-12',
         'country':'India','state':'Maharashtra','subscription_start_date':'2023-01-15','renewal_date':'2024-01-15',
         'subscription_type':'Paid','plan_type':'Standard','contract_type':'Annual','monthly_charges':14.99,
         'cltv':550,'churn_score':32,'csat_score':85,'complaint_count':0,'complaint_date':'','escalations':'N'},
        {'customerid':'CUST-0092','customer_name':'Aisha Thapa','gender':'Female','dob':'1995-11-20',
         'country':'Nepal','state':'Kathmandu','subscription_start_date':'2023-08-01','renewal_date':'2024-02-01',
         'subscription_type':'Organic','plan_type':'Basic','contract_type':'Monthly','monthly_charges':9.99,
         'cltv':210,'churn_score':82,'csat_score':30,'complaint_count':3,'complaint_date':'2024-01-10','escalations':'Y'},
        {'customerid':'CUST-0093','customer_name':'Priyesh Sen','gender':'Male','dob':'1964-07-05',
         'country':'India','state':'West Bengal','subscription_start_date':'2021-05-10','renewal_date':'2024-05-10',
         'subscription_type':'Refferal','plan_type':'Premium','contract_type':'Annual','monthly_charges':24.99,
         'cltv':1500,'churn_score':55,'csat_score':68,'complaint_count':1,'complaint_date':'2023-12-15','escalations':'N'},
    ])
    return sample.to_csv(index=False)


model   = load_model()
df_data = load_dataset()
df_importance = load_feature_importance()


# ----------------- RECOMMENDATIONS -----------------
def get_recommendation_details(probability):
    if probability >= 0.70:
        return {
            "level": "High", "badge_class": "badge-high", "color": "#e74c3c",
            "rec_title": "🚨 Urgent Retention Actions Required:",
            "points": [
                "**Immediate Outreach**: Initiate contact via their account manager or support team within 24 hours.",
                "**Offer Financial Incentives**: Provide a customized retention discount (e.g., 20% off for 3 months).",
                "**Dedicated Support**: Route all support inquiries from this customer directly to senior specialists.",
                "**Root Cause Investigation**: Review recent escalations and complaints to resolve outstanding service issues."
            ]
        }
    elif probability >= 0.40:
        return {
            "level": "Medium", "badge_class": "badge-medium", "color": "#f1c40f",
            "rec_title": "⚠️ Proactive Nurturing Recommended:",
            "points": [
                "**Personalized Survey**: Send a quick feedback request to identify pain points.",
                "**Feature Walkthrough**: Send targeted content highlighting underutilized platform features.",
                "**Standard Loyalty Reward**: Offer a minor value-add (e.g., a free temporary add-on).",
                "**Monitor Closely**: Flag account for monthly review and track changes in complaint counts."
            ]
        }
    else:
        return {
            "level": "Low", "badge_class": "badge-low", "color": "#2ecc71",
            "rec_title": "✅ Regular Engagement Actions:",
            "points": [
                "**Continue Regular Engagement**: Maintain scheduled marketing newsletters and standard updates.",
                "**Promote Long-term Commitment**: Prompt for annual contract renewal at their next billing cycle.",
                "**Value Check-ins**: Ensure they are receiving standard value from the platform."
            ]
        }


# ----------------- FEATURE ENGINEERING -----------------
def preprocess_uploaded_data(uploaded_df):
    X = uploaded_df.copy()
    X.columns = [c.lower().strip().replace(' ', '_') for c in X.columns]
    X = X.rename(columns={
        'subscription_start': 'subscription_start_date',
        'renewal': 'renewal_date',
        'complaint': 'complaint_date',
        'age': 'customer_age',
        'duration': 'subscription_duration_months',
        'monthly': 'monthly_charges'
    })
    for col in ['subscription_start_date', 'renewal_date', 'dob', 'complaint_date']:
        if col in X.columns:
            X[col] = pd.to_datetime(X[col], errors='coerce')
    ref = pd.Timestamp.today().normalize()
    if 'dob' in X.columns and 'customer_age' not in X.columns:
        X['customer_age'] = ((ref - X['dob']).dt.days / 365.25).round(1)
    if {'subscription_start_date', 'renewal_date'}.issubset(X.columns) and 'subscription_duration_months' not in X.columns:
        X['subscription_duration_months'] = ((X['renewal_date'] - X['subscription_start_date']).dt.days / 30.44).round(1)
    if 'complaint_date' in X.columns and 'days_since_last_complaint' not in X.columns:
        X['days_since_last_complaint'] = (ref - X['complaint_date']).dt.days
    expected = [
        'subscription_type', 'plan_type', 'contract_type', 'monthly_charges', 'cltv',
        'churn_score', 'country', 'state', 'gender', 'escalations', 'csat_score',
        'complaint_count', 'customer_age', 'subscription_duration_months', 'days_since_last_complaint'
    ]
    for col in expected:
        if col not in X.columns:
            X[col] = np.nan
    return X[expected]


# ----------------- SIDEBAR -----------------
st.sidebar.markdown('<div class="brand-title">🔮 ChurnLens</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="brand-subtitle">Customer Retention Portal</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")
navigation = st.sidebar.radio("Navigation Menu", ["Home", "Predict Churn", "About"])


# =============================================================================
# HOME PAGE
# =============================================================================
if navigation == "Home":
    st.markdown('<h1 class="main-title">Customer Churn Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Interactive dashboard containing project overviews, machine learning metrics, and dataset summaries.</p>', unsafe_allow_html=True)

    col_lead, col_metrics = st.columns([3, 2])
    with col_lead:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📋 Project Overview")
        st.markdown("""
        Customer churn represents a critical leakage in business revenue and customer lifetime value.
        This end-to-end analytical project focuses on identifying early indicators of customer churn
        to execute proactive retention strategies.

        **Key Goals:**
        * **Identify Risk Patterns**: Uncover the support and billing characteristics that drive customers to leave.
        * **Predictive Outreach**: Build a classifier capable of flagging high-risk customers before their renewal window.
        * **Actionable Retention**: Deliver specific retention recommendations tailored to individual customer risk levels.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Model Information")
        st.markdown("""
        The predictive model is a **Logistic Regression Pipeline** optimized for customer retention.
        * **Preprocessing**: Integrates numerical scaling (**StandardScaler**), categorical encoding (**OneHotEncoder**), and median/mode imputation (**SimpleImputer**).
        * **Class Balancing**: Balanced weights (**class_weight='balanced'**) are applied to handle the dataset's minority churn class effectively.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_metrics:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Model Performance (Test Set)")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown('<div class="custom-metric"><div class="custom-metric-title">Accuracy</div><div class="custom-metric-value custom-metric-accent">71.4%</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
            st.markdown('<div class="custom-metric"><div class="custom-metric-title">Precision</div><div class="custom-metric-value">50.0%</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="custom-metric"><div class="custom-metric-title">ROC-AUC</div><div class="custom-metric-value custom-metric-accent">81.3%</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
            st.markdown('<div class="custom-metric"><div class="custom-metric-title">Recall (Churn)</div><div class="custom-metric-value">76.7%</div></div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.85rem;color:var(--text-color);opacity:0.6;margin-top:15px;font-style:italic;text-align:center;">High Recall (76.7%) is prioritized to capture as many potential churners as possible.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if df_importance is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📈 Top Features Influencing Churn")
        dfi = df_importance.sort_values(by="Absolute Value", ascending=True).tail(12)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=dfi["Feature"], x=dfi["Coefficient"], orientation='h',
            marker=dict(color=np.where(dfi["Coefficient"] >= 0, '#e74c3c', '#2ecc71'), line=dict(width=0))
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='gray'),
            margin=dict(l=20, r=20, t=10, b=10), height=350,
            xaxis=dict(title="Model Coefficient (Positive = Increases Churn Risk)", gridcolor='rgba(128,128,128,0.1)', zerolinecolor='rgba(128,128,128,0.3)'),
            yaxis=dict(gridcolor='rgba(128,128,128,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if df_data is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📁 Dataset Summary & Insights")
        ds1, ds2, ds3, ds4 = st.columns(4)
        with ds1: st.metric("Total Historical Customers", len(df_data))
        with ds2: st.metric("Overall Dataset Churn Rate", f"{df_data['churn_flag'].mean()*100:.1f}%")
        with ds3: st.metric("Avg Monthly Charges", f"${df_data['monthly_charges'].mean():.2f}")
        with ds4: st.metric("Avg CLTV Value", f"${df_data['cltv'].mean():.1f}")
        st.write("")

        p1, p2 = st.columns(2)
        with p1:
            st.markdown("##### Churn Count by Contract Type")
            fig_c = px.histogram(df_data, x="contract_type", color="churn_flag", barmode="group",
                                 color_discrete_map={0:'#2ecc71',1:'#e74c3c'},
                                 labels={"churn_flag":"Churn Status","contract_type":"Contract Type"})
            fig_c.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='gray'),
                                 xaxis=dict(gridcolor='rgba(128,128,128,0.1)'), yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                                 legend=dict(title="Churn (1=Yes, 0=No)"), margin=dict(t=10))
            st.plotly_chart(fig_c, use_container_width=True)
        with p2:
            st.markdown("##### Churn Count by Plan Type")
            fig_p = px.histogram(df_data, x="plan_type", color="churn_flag", barmode="group",
                                 color_discrete_map={0:'#2ecc71',1:'#e74c3c'},
                                 labels={"churn_flag":"Churn Status","plan_type":"Plan Type"})
            fig_p.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='gray'),
                                 xaxis=dict(gridcolor='rgba(128,128,128,0.1)'), yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                                 legend=dict(title="Churn (1=Yes, 0=No)"), margin=dict(t=10))
            st.plotly_chart(fig_p, use_container_width=True)

        st.markdown('<div style="background-color:rgba(128,128,128,0.03);border-radius:8px;padding:20px;border:1px solid rgba(128,128,128,0.1);margin-top:20px;">', unsafe_allow_html=True)
        st.markdown("#### 💡 Strategic Business Insights from Given Dataset")
        st.markdown("""
        * **1. High Monthly Contract Churn**: Monthly contracts churn at **34.1%**, compared with **20.2%** for Annual contracts. *Action*: Offer **15% discount** or **1 month free** for switching to Annual.
        * **2. Support Escalation Impact**: Escalated customers churn at **62.1%**, compared with **24.0%** for non-escalated customers. *Action*: Establish a "Priority Retention Support" team to resolve escalations within 24 hours.
        * **3. Basic Plan Attrition**: Basic plan customers have the highest churn rate at **40.4%**, while Premium customers have the lowest churn rate at **18.9%**. *Action*: Review Basic plan pricing, content value, and onboarding.
        * **4. Acquisition Channel Review**: Paid customers show the lowest churn rate at **26.7%**, while Referral customers churn at **30.6%** in this dataset. *Action*: Review referral quality and post-signup engagement before expanding incentives.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")
        with st.expander("🔍 View Sample Cleaned Data"):
            st.dataframe(df_data.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PREDICT CHURN PAGE
# =============================================================================
elif navigation == "Predict Churn":
    st.markdown('<h1 class="main-title">🔮 Customer Churn Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predict individual customer churn using manual input, or process a batch using CSV upload.</p>', unsafe_allow_html=True)

    # ---- Session State Init ----
    for key, default in [
        ('batch_predictions', None),
        ('uploaded_df_cache', None),
        ('last_uploaded_file_id', None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    if model is None:
        st.error("The predictive model could not be loaded. Please ensure the model file is in the models folder.")
    else:
        tab_manual, tab_batch = st.tabs(["👤 Manual Input Prediction", "📤 Batch CSV Prediction"])

        # ----------------------------------------------------------------
        # MANUAL TAB
        # ----------------------------------------------------------------
        with tab_manual:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Input Customer Attributes")
            with st.form("manual_churn_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**👤 Personal & Location**")
                    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
                    customer_age = st.slider("Customer Age", min_value=18, max_value=100, value=40)
                    country = st.selectbox("Country", ["India", "Nepal"], index=0)
                    states_india = ['Maharashtra','Karnataka','Delhi','Nagaland','Meghalaya','Rajasthan','Telangana','Uttar Pradesh','Tamil Nadu','Gujarat','West Bengal']
                    state = st.selectbox("State", states_india if country == "India" else ['Kathmandu'])
                    subscription_type = st.selectbox("Subscription Type", ["Refferal", "Paid", "Organic"], index=1)
                with col2:
                    st.markdown("**💳 Subscription & Billing**")
                    plan_type = st.selectbox("Plan Type", ["Standard", "Premium", "Basic"], index=0)
                    contract_type = st.selectbox("Contract Type", ["Annual", "Monthly"], index=1)
                    monthly_charges = st.number_input("Monthly Charges ($)", min_value=6.99, max_value=200.00, value=13.99, step=1.0)
                    cltv = st.number_input("CLTV (Customer Lifetime Value)", min_value=30, max_value=5000, value=418, step=50)
                    subscription_duration_months = st.slider("Subscription Duration (Months)", min_value=0, max_value=120, value=12)
                with col3:
                    st.markdown("**📞 Experience & Support**")
                    churn_score = st.slider("Churn Score (Internal Indicator)", min_value=1, max_value=100, value=55)
                    csat_score = st.slider("CSAT Score (Customer Satisfaction)", min_value=10, max_value=100, value=53)
                    complaint_count = st.slider("Complaint Count", min_value=0, max_value=5, value=1)
                    escalations = st.selectbox("Escalations Raised?", ["Y", "N"], index=1)
                    days_since_last_complaint = st.number_input("Days Since Last Complaint", min_value=0, max_value=1000, value=30, step=10)
                submit_btn = st.form_submit_button("Predict Churn Risk", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if submit_btn:
                input_data = pd.DataFrame([{
                    'subscription_type': subscription_type, 'plan_type': plan_type,
                    'contract_type': contract_type, 'monthly_charges': monthly_charges,
                    'cltv': cltv, 'churn_score': churn_score, 'country': country,
                    'state': state, 'gender': gender, 'escalations': escalations,
                    'csat_score': csat_score, 'complaint_count': complaint_count,
                    'customer_age': customer_age,
                    'subscription_duration_months': subscription_duration_months,
                    'days_since_last_complaint': days_since_last_complaint
                }])
                try:
                    prob = model.predict_proba(input_data)[0, 1]
                    pred = model.predict(input_data)[0]
                    rec  = get_recommendation_details(prob)

                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("🎯 Prediction Output")
                    r1, r2 = st.columns(2)
                    with r1:
                        status_txt = "Churn Warning" if pred == 1 else "Retained Customer"
                        color_val  = "#e74c3c" if pred == 1 else "#2ecc71"
                        st.markdown(f'<div style="background-color:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);border-radius:10px;padding:20px;text-align:center;"><div style="font-size:0.85rem;color:var(--text-color);opacity:0.7;text-transform:uppercase;font-weight:600;">Model Prediction</div><div style="font-size:2.2rem;font-weight:700;color:{color_val};margin-top:8px;">{status_txt}</div></div>', unsafe_allow_html=True)
                    with r2:
                        st.markdown(f'<div style="background-color:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);border-radius:10px;padding:20px;text-align:center;"><div style="font-size:0.85rem;color:var(--text-color);opacity:0.7;text-transform:uppercase;font-weight:600;">Churn Probability</div><div style="font-size:2.2rem;font-weight:700;color:#FF4B4B;margin-top:8px;">{prob*100:.1f}%</div></div>', unsafe_allow_html=True)
                    st.write("")
                    ic, dc, rc = st.columns([1, 1, 1.5])
                    with ic:
                        st.markdown(f'<div class="risk-badge {rec["badge_class"]}">Risk Level: {rec["level"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div style="padding:10px 0;"><div class="info-item"><span class="info-label">Profile:</span> <span class="info-value">{customer_age} y/o {gender}</span></div><div class="info-item"><span class="info-label">Location:</span> <span class="info-value">{state}, {country}</span></div><div class="info-item"><span class="info-label">Plan:</span> <span class="info-value">{plan_type} ({contract_type})</span></div><div class="info-item"><span class="info-label">Monthly:</span> <span class="info-value">${monthly_charges:.2f}</span></div><div class="info-item"><span class="info-label">CLTV:</span> <span class="info-value">${cltv:.1f}</span></div></div>', unsafe_allow_html=True)
                    with dc:
                        st.markdown("**🔍 Model Metrics Details**")
                        st.markdown(f"* **Decision Probability**: `{prob:.4f}`\n* **Classification Class**: `{pred}` (Threshold 0.5)\n* **CSAT Score Given**: `{csat_score}/100`\n* **Complaint Status**: `{complaint_count}` complaints ({'Escalated' if escalations=='Y' else 'No escalation'})\n* **Days Since Complaint**: `{days_since_last_complaint}` days")
                    with rc:
                        st.markdown(f"<div class='recommendation-header' style='color:{rec['color']}'>{rec['rec_title']}</div>", unsafe_allow_html=True)
                        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                        for pt in rec['points']:
                            st.markdown(f"- {pt}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as ex:
                    st.error(f"Error executing model prediction: {ex}")

        # ----------------------------------------------------------------
        # BATCH TAB
        # ----------------------------------------------------------------
        with tab_batch:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Upload Customer Dataset")
            st.markdown("""
            Upload a CSV containing customer attributes. The system will automatically apply the required
            feature engineering (date parsing, age calculation, etc.) before running predictions.
            """)

            # Template download
            st.download_button(
                label="📥 Download Upload Template (Sample CSV)",
                data=get_sample_csv_template(),
                file_name="churn_upload_template.csv",
                mime="text/csv"
            )

            st.write("")
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
            st.markdown('</div>', unsafe_allow_html=True)

            # ── File change detection ────────────────────────────────────
            # We identify the file by (name, size) so the ID is stable across
            # every rerun (including the one triggered by clicking Download).
            # The raw file object is only read ONCE when a new file is detected.
            current_file_id = (uploaded_file.name, uploaded_file.size) if uploaded_file else None

            if current_file_id != st.session_state['last_uploaded_file_id']:
                # A different (or cleared) file → reset everything and read fresh
                st.session_state['last_uploaded_file_id'] = current_file_id
                st.session_state['batch_predictions']     = None
                if uploaded_file is not None:
                    try:
                        st.session_state['uploaded_df_cache'] = pd.read_csv(uploaded_file)
                    except Exception as e:
                        st.error(f"Could not read the uploaded CSV: {e}")
                        st.session_state['uploaded_df_cache'] = None
                else:
                    st.session_state['uploaded_df_cache'] = None

            # From here on, use only the cached copy — never re-read the file object
            df_upload = st.session_state['uploaded_df_cache']

            # ── Upload preview & charts ─────────────────────────────────
            if df_upload is not None:
                st.success("✅ File uploaded successfully!")

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("📈 Uploaded Dataset Insights")
                uc1, uc2 = st.columns(2)

                with uc1:
                    st.markdown("##### Distribution of Monthly Charges")
                    charge_col = next((c for c in df_upload.columns if c.lower().strip().replace(' ','_') in ['monthly_charges','monthly','monthlycharges']), None)
                    if charge_col and pd.api.types.is_numeric_dtype(df_upload[charge_col]):
                        fig_ch = px.histogram(df_upload, x=charge_col, nbins=20, color_discrete_sequence=['#FF4B4B'])
                        fig_ch.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='gray'),
                                             xaxis=dict(title="Charges ($)", gridcolor='rgba(128,128,128,0.1)'),
                                             yaxis=dict(title="Customer Count", gridcolor='rgba(128,128,128,0.1)'), margin=dict(t=10))
                        st.plotly_chart(fig_ch, use_container_width=True)
                    else:
                        st.info("Upload a file with a numeric 'monthly_charges' column to visualize.")

                with uc2:
                    st.markdown("##### Customer Distribution by Plan Type")
                    plan_col = next((c for c in df_upload.columns if c.lower().strip().replace(' ','_') in ['plan_type','plan','plantype']), None)
                    if plan_col:
                        vc = df_upload[plan_col].value_counts().reset_index()
                        vc.columns = ['Plan Type', 'Count']
                        fig_pl = px.bar(vc, x='Plan Type', y='Count', color='Plan Type', color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig_pl.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='gray'),
                                             xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                                             yaxis=dict(title="Customer Count", gridcolor='rgba(128,128,128,0.1)'),
                                             margin=dict(t=10), showlegend=False)
                        st.plotly_chart(fig_pl, use_container_width=True)
                    else:
                        st.info("Upload a file with a 'plan_type' column to visualize.")

                st.write("")
                st.markdown("**Preview of Uploaded Data (First 5 Rows):**")
                st.dataframe(df_upload.head(5), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # ── Run Batch Prediction ─────────────────────────────────
                if st.button("▶ Run Batch Prediction", use_container_width=True):
                    try:
                        with st.spinner("Processing features and predicting..."):
                            features_df = preprocess_uploaded_data(df_upload)
                            probs = model.predict_proba(features_df)[:, 1]
                            preds = model.predict(features_df)

                            df_res = df_upload.copy()
                            df_res['churn_probability']  = probs
                            df_res['churn_prediction']   = preds

                            risk_levels, recs = [], []
                            for p in probs:
                                r = get_recommendation_details(p)
                                risk_levels.append(r['level'])
                                recs.append("; ".join([
                                    pt.split('**: ')[1].replace('**','') if '**: ' in pt else pt.replace('**','')
                                    for pt in r['points']
                                ]))
                            df_res['churn_risk_level']           = risk_levels
                            df_res['actionable_recommendation']  = recs

                            # ✅ Store in session state — survives all reruns including download clicks
                            st.session_state['batch_predictions'] = df_res
                    except Exception as run_ex:
                        st.error(f"Prediction failed: {run_ex}")
                        import traceback
                        st.code(traceback.format_exc())

                # ── Results display ─────────────────────────────────────
                # This block lives OUTSIDE every try/except so the Download button
                # always renders with valid data no matter what triggered the rerun.
                if st.session_state['batch_predictions'] is not None:
                    df_res       = st.session_state['batch_predictions']
                    res_preds    = df_res['churn_prediction']
                    res_probs    = df_res['churn_probability']
                    res_risk     = df_res['churn_risk_level']
                    high_risk_df = df_res[res_risk == "High"]

                    st.write("---")
                    st.subheader("📊 Batch Analysis Summary")
                    b1, b2, b3 = st.columns(3)
                    with b1: st.metric("Total Customers Processed", len(df_res))
                    with b2: st.metric("Predicted Churn Rate", f"{(res_preds==1).mean()*100:.1f}%")
                    with b3:
                        hrc = int((res_risk == "High").sum())
                        st.metric("High Risk Customers", f"{hrc} ({hrc/len(df_res)*100:.1f}%)")
                    st.write("")

                    # Visualisations
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("📈 Prediction Visualizations")
                    vc1, vc2 = st.columns(2)
                    with vc1:
                        rc_counts = res_risk.value_counts().reset_index()
                        rc_counts.columns = ['Risk Level', 'Count']
                        fig_pie = px.pie(rc_counts, values='Count', names='Risk Level', color='Risk Level',
                                         color_discrete_map={'Low':'#2ecc71','Medium':'#f1c40f','High':'#e74c3c'},
                                         title="Distribution of Churn Risk Levels")
                        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='gray'))
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with vc2:
                        plan_col_r = next((c for c in df_res.columns if c.lower().strip().replace(' ','_') in ['plan_type','plan','plantype']), None)
                        if plan_col_r:
                            avg_p = df_res.groupby(plan_col_r)['churn_probability'].mean().reset_index()
                            avg_p.columns = ['Plan Type', 'Avg Churn Probability']
                            fig_bar = px.bar(avg_p, x='Plan Type', y='Avg Churn Probability', color='Plan Type',
                                             color_discrete_sequence=px.colors.qualitative.Pastel,
                                             title="Average Churn Probability by Plan Type")
                            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='gray'),
                                                   xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                                                   yaxis=dict(title="Avg Probability", gridcolor='rgba(128,128,128,0.1)'), showlegend=False)
                            st.plotly_chart(fig_bar, use_container_width=True)
                        else:
                            st.info("Plan type column not found.")

                    charge_col_r = next((c for c in df_res.columns if c.lower().strip().replace(' ','_') in ['monthly_charges','monthly','monthlycharges']), None)
                    if charge_col_r:
                        st.markdown("##### Churn Probability vs Monthly Charges by Risk Level")
                        hover_cols = ['customerid','customer_name'] if 'customer_name' in df_res.columns and 'customerid' in df_res.columns else []
                        fig_sc = px.scatter(df_res, x=charge_col_r, y='churn_probability', color='churn_risk_level',
                                            color_discrete_map={'Low':'#2ecc71','Medium':'#f1c40f','High':'#e74c3c'},
                                            hover_data=hover_cols,
                                            labels={'churn_probability':'Churn Probability', charge_col_r:'Monthly Charges ($)', 'churn_risk_level':'Risk Level'})
                        fig_sc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='gray'),
                                              xaxis=dict(gridcolor='rgba(128,128,128,0.1)'), yaxis=dict(gridcolor='rgba(128,128,128,0.1)'))
                        st.plotly_chart(fig_sc, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Insights
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("💡 Actionable Insights for this Batch")
                    num_h = int((res_risk == "High").sum())
                    num_m = int((res_risk == "Medium").sum())
                    st.markdown(f"* **Batch Health Summary**: Out of **{len(df_res)}** records, **{num_h} ({num_h/len(df_res)*100:.1f}%)** are **High Churn Risk** and **{num_m} ({num_m/len(df_res)*100:.1f}%)** are **Medium Churn Risk**.")
                    if len(high_risk_df) > 0:
                        if 'contract_type' in high_risk_df.columns:
                            mh = int((high_risk_df['contract_type'].astype(str).str.lower() == 'monthly').sum())
                            st.markdown(f"* **Contract Type Attrition**: **{mh}/{len(high_risk_df)} ({mh/len(high_risk_df)*100:.1f}%)** high-risk accounts are on **Monthly** contracts. *Action*: Target with 15% off switch-to-annual offer.")
                        if 'escalations' in high_risk_df.columns:
                            eh = int((high_risk_df['escalations'].astype(str).str.upper() == 'Y').sum())
                            st.markdown(f"* **Unresolved Escalations**: **{eh} ({eh/len(high_risk_df)*100:.1f}%)** high-risk customers have active escalated complaints. *Action*: Assign to Support Supervisor within 48 hours.")
                        if 'csat_score' in high_risk_df.columns:
                            lc = int((high_risk_df['csat_score'] < 60).sum())
                            st.markdown(f"* **Low CSAT**: **{lc} ({lc/len(high_risk_df)*100:.1f}%)** high-risk customers rated satisfaction below 60/100. *Action*: Send personalized survey or complimentary consultation.")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # High risk table
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("📋 Top High-Risk Customers")
                    disp = [c for c in ['customerid','customer_name','monthly_charges','cltv','churn_probability','churn_risk_level'] if c in high_risk_df.columns]
                    if disp:
                        st.dataframe(high_risk_df[disp].sort_values('churn_probability', ascending=False).head(10), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # ── Download ─────────────────────────────────────────
                    # csv_export is computed here, at render time, from session state.
                    # Clicking this button triggers a rerun but session_state['batch_predictions']
                    # is still populated, so the button re-renders with the same data → download works.
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("📋 Complete Prediction Output")
                    st.dataframe(df_res, use_container_width=True)

                    csv_export = df_res.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Complete Predictions CSV",
                        data=csv_export,
                        file_name="churn_predictions_export.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# ABOUT PAGE
# =============================================================================
elif navigation == "About":
    st.markdown('<h1 class="main-title">🔮 About ChurnLens Project</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Technical metadata, modeling specifications, and project scope.</p>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("💡 Analytical Scope")
    st.markdown("""
    This customer churn analysis represents a core business intelligence and machine learning initiative.
    By compiling relational database tables containing demographic, billing, and support escalations data,
    we map out the comprehensive journey of a customer.

    A Logistic Regression model is chosen for its transparency (coefficients mapping directly to positive
    and negative feature correlations) and fast inference speeds. This makes it ideal for deployment
    in low-latency real-time applications.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    ca1, ca2 = st.columns(2)
    with ca1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Numerical Features (8)")
        st.markdown("""
        * **monthly_charges**: Subscription recurring cost.
        * **cltv**: Customer Lifetime Value rating.
        * **churn_score**: Internal survey indicating attrition likelihood.
        * **csat_score**: Customer satisfaction score (10 to 100).
        * **complaint_count**: Count of service tickets (1 to 3).
        * **customer_age**: Age in years (calculated from DOB).
        * **subscription_duration_months**: Calculated subscription active period.
        * **days_since_last_complaint**: Time elapsed since last issue.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with ca2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🏷️ Categorical Features (7)")
        st.markdown("""
        * **subscription_type**: Referral, Paid, Organic.
        * **plan_type**: Standard, Premium, Basic.
        * **contract_type**: Monthly, Annual.
        * **country**: India, Nepal.
        * **state**: Customer location state.
        * **gender**: Male, Female.
        * **escalations**: Support complaint escalated (Y/N).

        **Methodology:**
        * Numerical → **StandardScaler** + **SimpleImputer** (median).
        * Categorical → **OneHotEncoder** + **SimpleImputer** (most_frequent).
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="text-align:center;margin-top:50px;font-size:0.9rem;color:var(--text-color);opacity:0.6;">ChurnLens App • Built in Streamlit • Deployment Ready</div>', unsafe_allow_html=True)
