import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, confusion_matrix
)
from fpdf import FPDF
from datetime import datetime
import os

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── 1. PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Risk Analyzer",
    page_icon="🏦",
    layout="centered",        # Mobile-first focus
    initial_sidebar_state="collapsed"
)

# ── 2. CSS DESIGN SYSTEM ──────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global Reset ── */
[data-testid="stAppViewContainer"] { background: #0b0f19; color: #f8fafc; }
[data-testid="stMainBlockContainer"] { max-width: 440px; padding: 0 1rem 80px; margin: 0 auto; }
[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"] { display: none; }
#MainMenu, footer, [data-testid="stToolbar"] { display: none; }

/* ── Top Nav Bar ── */
.top-nav {
    background: #111827;
    padding: 14px 20px 12px;
    margin: 0 -1rem 1.5rem;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 999;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.nav-left { display: flex; align-items: center; gap: 12px; }
.nav-right { display: flex; align-items: center; gap: 8px; }
.logo-tile {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #4CAF50, #2196F3);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.nav-title { font-size: 15px; font-weight: 600; color: #fff; margin: 0; }

/* ── Custom Cards ── */
.m-card {
    background: #1e293b;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 12px;
}
.m-card h3 {
    font-size: 13px; font-weight: 500; color: #94a3b8;
    margin: 0 0 12px; display: flex; align-items: center; gap: 6px;
}

/* ── Result Cards ── */
.risk-card { border-radius: 16px; padding: 24px; text-align: center; margin-bottom: 16px; }
.risk-card.low  { background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); color: #34d399; }
.risk-card.mid  { background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); color: #fbbf24; }
.risk-card.high { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); color: #f87171; }

/* ── Inputs & Buttons ── */
.stButton > button {
    width: 100%; border-radius: 12px; font-size: 14px; font-weight: 600;
    padding: 12px; background: #3b82f6; border: none; color: #fff; transition: 0.2s;
}
.stButton > button:hover { background: #2563eb; transform: translateY(-1px); }
.stSelectbox > div > div { border-radius: 12px; background: #1e293b !important; }

/* ── Wizard UI ── */
.wizard-progress { height: 6px; background: #111827; border-radius: 10px; margin-bottom: 24px; overflow: hidden; }
.wizard-bar { height: 100%; background: #4CAF50; transition: 0.4s ease; }
.step-label { font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# ── 3. LOCALIZATION (STRINGS) ────────────────────────────────────────────────
STRINGS = {
  "en": {
    "hero_title":    "Check if you qualify for a loan",
    "hero_sub":      "Answer 5 simple questions. Takes 2 minutes.",
    "cta":           "Start my assessment",
    "q1_label":      "What is your monthly income? (KSh)",
    "q2_label":      "How much do you want to borrow? (KSh)",
    "q3_label":      "How long have you been employed? (years)",
    "q4_label":      "What is your current monthly debt? (KSh)",
    "q5_label":      "Any defaults or late payments in the last 2 years?",
    "result_low":    "Great news! You are likely to qualify.",
    "result_mid":    "You may qualify, but lenders may want more info.",
    "result_high":   "Your application carries some risk.",
    "tips_title":    "How to improve your chances",
    "download_btn":  "Download my report (PDF)",
    "next":          "Next →",
    "back":          "← Back",
    "step_of":       "Step {current} of {total}",
    "switch_analyst":"⚙ Analyst view",
    "switch_app":    "👤 Applicant view"
  },
  "sw": {
    "hero_title":    "Angalia kama unastahili mkopo",
    "hero_sub":      "Jibu maswali 5 rahisi. Inachukua dakika 2.",
    "cta":           "Anza tathmini yangu",
    "q1_label":      "Mapato yako ya kila mwezi ni kiasi gani? (KSh)",
    "q2_label":      "Unataka kukopa kiasi gani? (KSh)",
    "q3_label":      "Umefanya kazi kwa muda gani? (miaka)",
    "q4_label":      "Deni lako la kila mwezi ni kiasi gani? (KSh)",
    "q5_label":      "Je, umewahi kukosa malipo ya mkopo miaka 2 iliyopita?",
    "result_low":    "Habari njema! Una uwezekano mkubwa wa kupata mkopo.",
    "result_mid":    "Unaweza kupata mkopo, lakini wakopeshaji wanaweza kutaka maelezo zaidi.",
    "result_high":   "Ombi lako lina hatari fulani.",
    "tips_title":    "Unachoweza kufanya kuboresha nafasi zako",
    "download_btn":  "Pakua ripoti yangu (PDF)",
    "next":          "Ifuatayo →",
    "back":          "← Rudi",
    "step_of":       "Hatua {current} ya {total}",
    "switch_analyst":"⚙ Mtazamo wa Mchambuzi",
    "switch_app":    "👤 Mtazamo wa Mwombaji"
  }
}

# ── 3. PDF GENERATOR ────────────────────────────────────────────────────────
def generate_pdf(data, result_text, tips, lang):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 15, "Loan Eligibility Summary", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%d %b %Y %H:%M')}", ln=True, align="C")
    pdf.ln(10)
    
    # Inputs Table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Your Information:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for k, v in data.items():
        if k in ["monthly_income", "loan_amnt", "monthly_debt"]:
            pdf.cell(90, 8, f"{k.replace('_',' ').title()}:", border=1)
            pdf.cell(0, 8, f"KSh {v:,.0f}", border=1, ln=True)
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Result:", ln=True)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 15, result_text, ln=True, align="C")
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Recommended Next Steps:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for tip in tips:
        pdf.multi_cell(0, 8, f"- {tip}")
    
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 10, "Disclaimer: This is an automated assessment for demonstration purposes. Not financial advice.", align="C")
    
    return pdf.output(dest='S').encode('latin-1')

# ── 4. SHARED LOGIC & DATA ────────────────────────────────────────────────────
FEATURES = ["loan_amnt","int_rate","annual_inc","dti","emp_length",
            "open_acc","revol_util","pub_rec","delinq_2yrs",
            "loan_inc_ratio","int_inc_ratio"]

def generate_data(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "loan_amnt":   rng.integers(1000, 40000, n),
        "int_rate":    rng.uniform(5, 30, n).round(2),
        "annual_inc":  rng.integers(20000, 250000, n),
        "dti":         rng.uniform(0, 40, n).round(2),
        "emp_length":  rng.integers(0, 11, n),
        "open_acc":    rng.integers(1, 30, n),
        "revol_util":  rng.uniform(0, 100, n).round(2),
        "pub_rec":     rng.choice([0,1,2,3], n, p=[0.80,0.12,0.05,0.03]),
        "delinq_2yrs": rng.choice([0,1,2,3,4], n, p=[0.70,0.18,0.07,0.03,0.02]),
    })
    df["loan_inc_ratio"] = (df["loan_amnt"] / df["annual_inc"]).round(4)
    df["int_inc_ratio"]  = ((df["int_rate"]/100)*df["loan_amnt"]/df["annual_inc"]).round(4)
    log_odds = (-3.5 + 0.04*df["int_rate"] + 0.05*df["dti"] + 0.15*df["loan_inc_ratio"] + 0.6*df["pub_rec"] + 0.3*df["delinq_2yrs"] + 0.005*df["revol_util"] - 0.05*df["emp_length"] + rng.normal(0,0.5,n))
    prob = 1/(1+np.exp(-log_odds))
    df["loan_status"] = (rng.uniform(0,1,n) < prob).astype(int)
    return df

def load_data(n=2500, seed=42):
    try:
        df = pd.read_csv("data/processed/real_sample.csv")
        df["loan_inc_ratio"] = (df["loan_amnt"]/df["annual_inc"]).round(4)
        df["int_inc_ratio"]  = ((df["int_rate"]/100)*df["loan_amnt"]/df["annual_inc"]).round(4)
        return df.sample(min(n, len(df)), random_state=seed)
    except:
        return generate_data(n, seed)

@st.cache_resource(show_spinner="Training model...")
def train(_df, model_name, data_len):
    X = _df[FEATURES].values
    y = _df["loan_status"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    if model_name == "XGBoost" and XGB_AVAILABLE:
        clf = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric="logloss", random_state=42)
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:,1]
    fi = getattr(clf, "feature_importances_", np.ones(len(FEATURES)))
    shap_vals = None
    if SHAP_AVAILABLE and model_name in ["XGBoost","Random Forest"]:
        try:
            exp = shap.TreeExplainer(clf)
            shap_vals = exp.shap_values(X_te)
            if isinstance(shap_vals, list): shap_vals = shap_vals[1]
        except: pass
    return dict(clf=clf, X_te=X_te, y_te=y_te, y_pred=y_pred, y_prob=y_prob, acc=accuracy_score(y_te,y_pred), prec=precision_score(y_te,y_pred,zero_division=0), rec=recall_score(y_te,y_pred,zero_division=0), f1=f1_score(y_te,y_pred,zero_division=0), fi=fi, shap_vals=shap_vals)

# ── 5. SESSION STATE ──────────────────────────────────────────────────────────
if "mode" not in st.session_state: st.session_state.mode = "applicant"
if "tab" not in st.session_state: st.session_state.tab = "home"
if "df" not in st.session_state: st.session_state.df = load_data()
if "model_choice" not in st.session_state: st.session_state.model_choice = "XGBoost"
if "wizard_step" not in st.session_state: st.session_state.wizard_step = 0
if "wizard_data" not in st.session_state: st.session_state.wizard_data = {}
if "result_ready" not in st.session_state: st.session_state.result_ready = False
if "history" not in st.session_state: st.session_state.history = []
if "lang" not in st.session_state: st.session_state.lang = "en"

T = STRINGS[st.session_state.lang]
df = st.session_state.df
res = train(df, st.session_state.model_choice, len(df))

# ── 6. TOP NAV BAR ────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-nav">
  <div class="nav-left">
    <div class="logo-tile">🏦</div>
    <div class="nav-title">Loan Risk Analyzer</div>
  </div>
  <div class="nav-right" id="nav-right"></div>
</div>
""", unsafe_allow_html=True)

# Using columns for the interactive part of the nav (buttons)
nav_col1, nav_col2 = st.columns([1, 1])
with nav_col1:
    lang = st.radio("Lang", ["EN", "SW"], index=0 if st.session_state.lang == "en" else 1, horizontal=True, label_visibility="collapsed")
    if lang.lower() != st.session_state.lang:
        st.session_state.lang = lang.lower()
        st.rerun()

with nav_col2:
    if st.session_state.mode == "applicant":
        if st.button(T["switch_analyst"]):
            st.session_state.mode = "analyst"
            st.rerun()
    else:
        if st.button(T["switch_app"]):
            st.session_state.mode = "applicant"
            st.rerun()

# ── 7. MODE DISPATCHER ────────────────────────────────────────────────────────
if st.session_state.mode == "applicant":
    # ══════════════════════════════════════════════════════════════════════════
    # MODE A: APPLICANT VIEW
    # ══════════════════════════════════════════════════════════════════════════
    step = st.session_state.wizard_step
    
    if step == 0:
        # Hero Screen
        st.markdown(f"<h1 style='text-align: center; margin-top: 40px;'>{T['hero_title']}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: #94a3b8; font-size: 18px;'>{T['hero_sub']}</p>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(T["cta"]):
            st.session_state.wizard_step = 1
            st.session_state.result_ready = False
            st.rerun()
            
        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns(3)
        cols[0].markdown("<p style='text-align:center; font-size:12px; color:#64748b;'>🔒 Private</p>", unsafe_allow_html=True)
        cols[1].markdown("<p style='text-align:center; font-size:12px; color:#64748b;'>⚡ Instant result</p>", unsafe_allow_html=True)
        cols[2].markdown("<p style='text-align:center; font-size:12px; color:#64748b;'>🆓 Free</p>", unsafe_allow_html=True)

    elif 1 <= step <= 5:
        # Wizard Progress
        st.markdown(f"<div class='step-label'>{T['step_of'].format(current=step, total=5)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='wizard-progress'><div class='wizard-bar' style='width: {step*20}%'></div></div>", unsafe_allow_html=True)
        
        with st.container(border=True):
            if step == 1:
                st.session_state.wizard_data["monthly_income"] = st.number_input(T["q1_label"], 0, 10000000, st.session_state.wizard_data.get("monthly_income", 50000), 1000)
            elif step == 2:
                inc = st.session_state.wizard_data.get("monthly_income", 1)
                amt = st.number_input(T["q2_label"], 1000, 5000000, st.session_state.wizard_data.get("loan_amnt", 10000), 1000)
                st.session_state.wizard_data["loan_amnt"] = amt
                ratio = amt / inc
                st.info(f"That is {ratio:.1f}x your monthly income.")
                if ratio > 5: st.warning("⚠️ This is more than 5x your monthly income.")
            elif step == 3:
                opts = ["Less than 1 year","1–2 years","3–5 years","6–10 years","10+ years"]
                choice = st.select_slider(T["q3_label"], options=opts, value=st.session_state.wizard_data.get("emp_choice", "3–5 years"))
                st.session_state.wizard_data["emp_choice"] = choice
                mapping = {"Less than 1 year": 0, "1–2 years": 1, "3–5 years": 3, "6–10 years": 7, "10+ years": 10}
                st.session_state.wizard_data["emp_length"] = mapping[choice]
            elif step == 4:
                debt = st.number_input(T["q4_label"], 0, 5000000, st.session_state.wizard_data.get("monthly_debt", 5000), 500)
                st.session_state.wizard_data["monthly_debt"] = debt
                inc = st.session_state.wizard_data.get("monthly_income", 1)
                dti = (debt / inc) * 100
                st.info(f"Your debt ratio is {dti:.1f}% — ideal is below 35%.")
            elif step == 5:
                choice = st.radio(T["q5_label"], ["No","Yes — once","Yes — more than once"], index=st.session_state.wizard_data.get("delinq_choice_idx", 0))
                mapping = {"No": 0, "Yes — once": 1, "Yes — more than once": 2}
                st.session_state.wizard_data["delinq_choice_idx"] = ["No","Yes — once","Yes — more than once"].index(choice)
                st.session_state.wizard_data["delinq_2yrs"] = mapping[choice]

        col_l, col_r = st.columns(2)
        with col_l:
            if st.button(T["back"]):
                st.session_state.wizard_step -= 1
                st.rerun()
        with col_r:
            if st.button(T["next"]):
                if step < 5:
                    st.session_state.wizard_step += 1
                else:
                    st.session_state.result_ready = True
                    st.session_state.wizard_step = 6
                st.rerun()

    elif step == 6 and st.session_state.result_ready:
        # ── RESULT SCREEN ──
        d = st.session_state.wizard_data
        annual_inc = d["monthly_income"] * 12
        loan_amnt = d["loan_amnt"]
        dti = (d["monthly_debt"] * 12) / annual_inc * 100
        input_row = np.array([[loan_amnt, 14.0, annual_inc, dti, d["emp_length"], 3, 45.0, 0, d["delinq_2yrs"], loan_amnt/annual_inc, (0.14)*loan_amnt/annual_inc]])
        prob = res["clf"].predict_proba(input_row)[0][1]
        
        # Traffic Light Result
        if prob < 0.30:
            cls, icon, head, sub = "low", "✅", T["result_low"], "Based on your information, a lender is likely to approve your application."
            res_val = "Low"
        elif prob < 0.60:
            cls, icon, head, sub = "mid", "⚠️", T["result_mid"], "Your profile is acceptable but some lenders may ask for more info."
            res_val = "Medium"
        else:
            cls, icon, head, sub = "high", "🚨", T["result_high"], "Your current profile makes lenders cautious. See tips below."
            res_val = "High"

        st.markdown(f"""
        <div class="risk-card {cls}">
          <div style="font-size: 48px; margin-bottom: 10px;">{icon}</div>
          <h2 style="color: inherit; margin: 0;">{head}</h2>
          <p style="color: inherit; opacity: 0.8; margin-top: 10px; font-size: 14px;">{sub}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Personalised Tips
        st.markdown(f"<h3>{T['tips_title']}</h3>", unsafe_allow_html=True)
        tips = []
        if dti > 40: tips.append(f"Your debt is high. Try to pay off KSh {d['monthly_debt']*0.2:,.0f} before applying.")
        if (loan_amnt/annual_inc) > 3: tips.append(f"Loan is large for your income. Try borrowing KSh {loan_amnt*0.7:,.0f} instead.")
        if d["emp_length"] < 2: tips.append("Lenders like 2+ years of job stability. If possible, wait a few more months.")
        if d["delinq_2yrs"] > 0: tips.append("Past late payments hurt your score. Keep all current accounts up to date.")
        
        # Fill to 3 tips
        if len(tips) < 3: tips.append("Apply to a SACCO first — they often have better terms for first-time borrowers.")
        if len(tips) < 3: tips.append("Save 3 months of repayments in an emergency fund to show discipline.")
        
        for t in tips[:3]:
            st.markdown(f"<div class='m-card' style='padding: 12px 16px; margin-bottom: 8px;'>💡 {t}</div>", unsafe_allow_html=True)

        # Scenario Simulator (The ONLY ML part)
        st.markdown("<br><h4>What if I borrow a different amount?</h4>", unsafe_allow_html=True)
        sim_amt = st.slider("Loan Amount (KSh)", 1000, int(loan_amnt*2), int(loan_amnt), 5000, label_visibility="collapsed")
        sim_row = input_row.copy()
        sim_row[0,0] = sim_amt
        sim_row[0,9] = sim_amt / annual_inc
        sim_row[0,10] = (0.14) * sim_amt / annual_inc
        sim_prob = res["clf"].predict_proba(sim_row)[0][1]
        s_cls = "🟢 Low" if sim_prob < 0.3 else "🟠 Medium" if sim_prob < 0.6 else "🔴 High"
        st.markdown(f"<div style='text-align:center; font-weight:600;'>Risk level at KSh {sim_amt:,}: <span style='color:{'#34d399' if sim_prob<0.3 else '#fbbf24' if sim_prob<0.6 else '#f87171'}'>{s_cls}</span></div>", unsafe_allow_html=True)

        # PDF & Reset
        st.markdown("<br>", unsafe_allow_html=True)
        pdf_bytes = generate_pdf(d, head, tips[:3], st.session_state.lang)
        st.download_button(T["download_btn"], data=pdf_bytes, file_name="Loan_Assessment.pdf", mime="application/pdf", use_container_width=True)
        
        if st.button("Check another loan →"):
            st.session_state.wizard_step = 0
            st.session_state.result_ready = False
            st.session_state.wizard_data = {}
            st.rerun()
            
        # Log to history
        if "last_logged" not in st.session_state or st.session_state.last_logged != d:
            st.session_state.history.append({"date": datetime.now().strftime("%d %b %H:%M"), "loan_amnt": loan_amnt, "result": res_val, "monthly_income": d["monthly_income"]})
            st.session_state.last_logged = d.copy()

else:
    # ══════════════════════════════════════════════════════════════════════════
    # MODE B: ANALYST VIEW
    # ══════════════════════════════════════════════════════════════════════════
    tabs = st.columns(4)
    if tabs[0].button("🏠 Home"): st.session_state.tab = "home"
    if tabs[1].button("📂 Data"): st.session_state.tab = "data"
    if tabs[2].button("🔍 Predict"): st.session_state.tab = "predict"
    if tabs[3].button("🤖 Models"): st.session_state.tab = "models"
    
    st.divider()
    t = st.session_state.tab
    
    if t == "home":
        st.subheader("Model Performance Overview")
        # 2x2 Grid
        c1, c2 = st.columns(2)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Default Rate", f"{df['loan_status'].mean():.1%}")
        c3, c4 = st.columns(2)
        c3.metric("Model Accuracy", f"{res['acc']:.1%}")
        c4.metric("F1 Score", f"{res['f1']:.1%}")
        
        # Charts
        fi_df = pd.DataFrame({"Feature": FEATURES, "Importance": res["fi"]}).sort_values("Importance")
        fig = px.bar(fi_df.tail(8), x="Importance", y="Feature", orientation="h", title="Top Feature Importance", color_continuous_scale="Blues", color="Importance")
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        
        if st.session_state.history:
            st.markdown("### Recent Predictions")
            st.table(pd.DataFrame(st.session_state.history).tail(5))

    elif t == "data":
        st.subheader("Dataset Management")
        src = st.radio("Choose source", ["Sample dataset", "Upload CSV", "CSV URL"], horizontal=True, label_visibility="collapsed")
        
        if src == "Sample dataset":
            n_rows = st.slider("Rows to load", 500, 5000, 1500)
            if st.button("Refresh Data"):
                st.session_state.df = load_data(n_rows)
                st.rerun()
        elif src == "Upload CSV":
            up = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
            if up:
                try:
                    new_df = pd.read_csv(up)
                    st.session_state.df = new_df
                    st.success("File uploaded successfully!")
                except Exception as e: st.error(f"Error: {e}")
        
        # Stats Card
        st.markdown(f"""
        <div class='m-card'>
          <h3>📋 Dataset Stats</h3>
          <div style='display:flex; justify-content:space-between; font-size:13px;'>
            <span>Rows: <b>{len(df):,}</b></span>
            <span>Features: <b>{len(df.columns)-1}</b></span>
            <span>Default Rate: <b>{df['loan_status'].mean():.1%}</b></span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Explore Distribution"):
            feat = st.selectbox("Feature", FEATURES)
            fig_h = px.histogram(df, x=feat, color="loan_status", barmode="overlay", color_discrete_map={0:"#4CAF50",1:"#F44336"})
            fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=250)
            st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar":False})

    elif t == "predict":
        st.subheader("Model Inference")
        p_mode = st.radio("Method", ["Manual", "Batch", "Sample"], horizontal=True)
        
        if p_mode == "Manual":
            with st.form("technical_predict"):
                cols = st.columns(2)
                l_amt = cols[0].number_input("Loan Amount", 1000, 50000, 10000)
                i_rate = cols[1].number_input("Interest Rate", 5.0, 30.0, 14.0)
                ann_inc = cols[0].number_input("Annual Income", 10000, 500000, 60000)
                dti_v = cols[1].slider("DTI Ratio", 0.0, 100.0, 20.0)
                emp_v = cols[0].slider("Employment (Yrs)", 0, 20, 5)
                sub_p = st.form_submit_button("Run Prediction", use_container_width=True)
                if sub_p:
                    in_row = np.array([[l_amt, i_rate, ann_inc, dti_v, emp_v, 5, 45.0, 0, 0, l_amt/ann_inc, (i_rate/100)*l_amt/ann_inc]])
                    p_val = res["clf"].predict_proba(in_row)[0][1]
                    st.markdown(f"<div class='m-card' style='text-align:center;'><h4>Probability of Default</h4><h1 style='color:{'#f87171' if p_val > 0.5 else '#34d399'}'>{p_val:.2%}</h1></div>", unsafe_allow_html=True)

        elif p_mode == "Batch":
            up_b = st.file_uploader("Upload CSV for scoring")
            if up_b:
                b_df = pd.read_csv(up_b)
                # (Batch logic similar to original app)
                st.success("Ready for batch scoring.")

    elif t == "models":
        st.subheader("Model Diagnostics")
        choice = st.selectbox("Active Algorithm", ["XGBoost","Random Forest","Logistic Regression"], index=["XGBoost","Random Forest","Logistic Regression"].index(st.session_state.model_choice))
        if choice != st.session_state.model_choice:
            st.session_state.model_choice = choice
            st.rerun()
            
        st.metric("Test Accuracy", f"{res['acc']:.1%}", delta=f"{res['f1']:.1%} F1")
        
        # Performance Charts
        c1, c2 = st.columns(2)
        fpr, tpr, _ = roc_curve(res["y_te"], res["y_prob"])
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC: {auc(fpr,tpr):.3f}"))
        fig_r.update_layout(title="ROC Curve", height=250, margin=dict(t=30,b=0,l=0,r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        c1.plotly_chart(fig_r, use_container_width=True)
        
        cm = confusion_matrix(res["y_te"], res["y_pred"])
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
        fig_cm.update_layout(height=250, margin=dict(t=30,b=0,l=0,r=0), paper_bgcolor="rgba(0,0,0,0)")
        c2.plotly_chart(fig_cm, use_container_width=True)
        
        if SHAP_AVAILABLE and res["shap_vals"] is not None:
            st.markdown("### Global Feature Importance (SHAP)")
            # Using Mean Absolute SHAP values
            sh_df = pd.DataFrame({"Feature": FEATURES, "Impact": np.abs(res["shap_vals"]).mean(axis=0)}).sort_values("Impact")
            fig_sh = px.bar(sh_df, x="Impact", y="Feature", orientation="h", color="Impact", color_continuous_scale="Oranges")
            fig_sh.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_sh, use_container_width=True)