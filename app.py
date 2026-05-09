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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Risk Analyzer",
    page_icon="🏦",
    layout="centered",          # mobile-first: centered not wide
    initial_sidebar_state="collapsed"
)

# ── Mobile CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global reset ── */
[data-testid="stAppViewContainer"] { background: #0b0f19; color: #e2e8f0; }
.stMarkdown, .stText, p, span, div { color: #e2e8f0; }
[data-testid="stMainBlockContainer"] { max-width: 440px; padding: 0 0 80px; margin: 0 auto; }
[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"] { display: none; }

/* ── Top nav bar ── */
.top-nav {
    background: #111827;
    padding: 14px 20px 12px;
    margin: 0 -1rem 1rem;
    display: flex; align-items: center; gap: 12px;
    position: sticky; top: 0; z-index: 999;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.top-nav .logo {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #4CAF50, #2196F3);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.top-nav h1 { color: #fff; font-size: 17px; font-weight: 500; margin: 0; }
.top-nav p  { color: #94a3b8; font-size: 11px; margin: 0; }

/* ── Cards ── */
.m-card {
    background: #1e293b;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 12px;
}
.m-card h3 {
    font-size: 13px; font-weight: 500;
    color: #cbd5e1; margin: 0 0 12px;
    display: flex; align-items: center; gap: 6px;
}

/* ── Stat grid ── */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; }
.stat-tile {
    background: #1e293b; border-radius: 14px;
    padding: 12px 14px;
    border: 1px solid rgba(255,255,255,0.1);
}
.stat-tile .label { font-size: 11px; color: #94a3b8; margin-bottom: 4px; }
.stat-tile .value { font-size: 22px; font-weight: 500; color: #f8fafc; }
.stat-tile .value.green { color: #4ade80; }
.stat-tile .value.red   { color: #f87171; }

/* ── Upload zone ── */
.upload-zone {
    border: 1.5px dashed rgba(255,255,255,0.2);
    border-radius: 14px; padding: 28px 16px;
    text-align: center; background: #0f172a;
    margin-bottom: 10px;
}
.upload-zone .icon { font-size: 32px; color: #cbd5e1; margin-bottom: 8px; }
.upload-zone p { font-size: 13px; color: #94a3b8; margin-bottom: 6px; }
.upload-zone .hint { font-size: 11px; color: #64748b; }

/* ── Source selector ── */
.source-opt {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 14px;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px; margin-bottom: 8px;
    background: #1e293b; cursor: pointer;
}
.source-opt.active { border-color: #60a5fa; background: #0f172a; }
.source-opt .icon { font-size: 22px; }
.source-opt .title { font-size: 13px; font-weight: 500; color: #f8fafc; margin: 0; }
.source-opt .sub   { font-size: 11px; color: #94a3b8; margin: 0; }

/* ── Risk result ── */
.risk-result {
    border-radius: 16px; padding: 20px;
    text-align: center; margin-bottom: 12px;
}
.risk-result.low  { background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); }
.risk-result.mid  { background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); }
.risk-result.high { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); }
.risk-pct { font-size: 42px; font-weight: 500; }
.risk-pct.low  { color: #34d399; }
.risk-pct.mid  { color: #fbbf24; }
.risk-pct.high { color: #f87171; }
.risk-text { font-size: 13px; margin-top: 4px; }
.risk-text.low  { color: #6ee7b7; }
.risk-text.mid  { color: #fcd34d; }
.risk-text.high { color: #fca5a5; }

/* ── Data row ── */
.d-row {
    display: flex; justify-content: space-between;
    align-items: center; padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 13px;
}
.d-row:last-child { border-bottom: none; }
.d-row .lbl { color: #94a3b8; }
.d-row .val { color: #f8fafc; font-weight: 500; }

/* ── Badge ── */
.badge {
    display: inline-block; font-size: 11px; font-weight: 500;
    padding: 3px 9px; border-radius: 20px;
}
.badge.green { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; }
.badge.red   { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }
.badge.blue  { background: rgba(59, 130, 246, 0.2); color: #93c5fd; }
.badge.amber { background: rgba(245, 158, 11, 0.2); color: #fcd34d; }

/* ── Bottom tab bar ── */
.bottom-bar {
    position: fixed; bottom: 0; left: 50%; transform: translateX(-50%);
    width: 100%; max-width: 440px;
    background: #111827; border-top: 1px solid rgba(255,255,255,0.1);
    display: flex; padding: 8px 0 16px; z-index: 998;
}
.tab-btn {
    flex: 1; text-align: center; cursor: pointer;
    font-size: 10px; color: #64748b;
    display: flex; flex-direction: column; align-items: center; gap: 3px;
}
.tab-btn.active { color: #f8fafc; }
.tab-btn .icon { font-size: 22px; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"] { display: none; }
.stSelectbox > div > div { border-radius: 12px; background: #1e293b; color: #f8fafc; border-color: rgba(255,255,255,0.1); }
.stNumberInput > div > div > input { color: #f8fafc; }
.stSlider > div > div > div > div { color: #f8fafc; }
.stButton > button {
    width: 100%; border-radius: 12px;
    font-size: 14px; font-weight: 500;
    background: #3b82f6; color: #fff;
    border: none; padding: 12px;
}
.stButton > button:hover { background: #2563eb; color: #fff; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
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
    log_odds = (
        -3.5 + 0.04*df["int_rate"] + 0.05*df["dti"]
        + 0.15*df["loan_inc_ratio"] + 0.6*df["pub_rec"]
        + 0.3*df["delinq_2yrs"] + 0.005*df["revol_util"]
        - 0.05*df["emp_length"] + rng.normal(0,.5,n)
    )
    prob = 1/(1+np.exp(-log_odds))
    df["loan_status"] = (rng.uniform(0,1,n) < prob).astype(int)
    return df

FEATURES = ["loan_amnt","int_rate","annual_inc","dti","emp_length",
            "open_acc","revol_util","pub_rec","delinq_2yrs",
            "loan_inc_ratio","int_inc_ratio"]

@st.cache_resource(show_spinner="Training model…")
def train(df_hash, model_name):
    df = generate_data()
    X = df[FEATURES].values
    y = df["loan_status"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    if model_name == "XGBoost" and XGB_AVAILABLE:
        clf = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                            use_label_encoder=False, eval_metric="logloss", random_state=42)
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
    return dict(clf=clf, X_te=X_te, y_te=y_te, y_pred=y_pred, y_prob=y_prob,
                acc=accuracy_score(y_te,y_pred),
                prec=precision_score(y_te,y_pred,zero_division=0),
                rec=recall_score(y_te,y_pred,zero_division=0),
                f1=f1_score(y_te,y_pred,zero_division=0),
                fi=fi, shap_vals=shap_vals)

# ── Session state ─────────────────────────────────────────────────────────────
if "tab" not in st.session_state:   st.session_state.tab = "home"
if "df"  not in st.session_state:   st.session_state.df  = generate_data()
if "model_choice" not in st.session_state: st.session_state.model_choice = "XGBoost"

df = st.session_state.df
res = train(len(df), st.session_state.model_choice)

# ── Top nav ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-nav">
  <div class="logo">🏦</div>
  <div>
    <h1>Loan Risk Analyzer</h1>
    <p>ML-powered credit assessment</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tab buttons ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🏠 Home"):    st.session_state.tab = "home"
with col2:
    if st.button("📂 Data"):    st.session_state.tab = "data"
with col3:
    if st.button("🔍 Predict"): st.session_state.tab = "predict"
with col4:
    if st.button("🤖 Models"):  st.session_state.tab = "models"

tab = st.session_state.tab
st.divider()


# ════════════════════════════════════════════════════════════════════════════
# HOME TAB
# ════════════════════════════════════════════════════════════════════════════
if tab == "home":

    dr = df["loan_status"].mean()
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-tile"><div class="label">Total records</div><div class="value">{len(df):,}</div></div>
      <div class="stat-tile"><div class="label">Default rate</div><div class="value red">{dr:.1%}</div></div>
      <div class="stat-tile"><div class="label">Model accuracy</div><div class="value green">{res['acc']:.1%}</div></div>
      <div class="stat-tile"><div class="label">F1 score</div><div class="value green">{res['f1']:.1%}</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="m-card">
      <h3>📊 Active model — {st.session_state.model_choice}</h3>
      <div class="d-row"><span class="lbl">Precision</span><span class="val">{res['prec']:.1%}</span></div>
      <div class="d-row"><span class="lbl">Recall</span><span class="val">{res['rec']:.1%}</span></div>
      <div class="d-row"><span class="lbl">F1 score</span><span class="val">{res['f1']:.1%}</span></div>
      <div class="d-row"><span class="lbl">Test set size</span><span class="val">{len(res['X_te'])}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Feature importance chart
    fi_df = pd.DataFrame({"Feature": FEATURES, "Importance": res["fi"]}).sort_values("Importance")
    fig = px.bar(fi_df.tail(8), x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues")
    fig.update_layout(height=280, margin=dict(t=10,b=10,l=0,r=0),
                      showlegend=False, coloraxis_showscale=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    st.markdown('<div class="m-card"><h3>📈 Top risk factors</h3>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # Default distribution
    fig2 = px.pie(values=df["loan_status"].value_counts(),
                  names=["No Default","Default"], hole=0.5,
                  color_discrete_map={"No Default":"#4CAF50","Default":"#F44336"})
    fig2.update_layout(height=240, margin=dict(t=10,b=10,l=0,r=0),
                       paper_bgcolor="rgba(0,0,0,0)")
    st.markdown('<div class="m-card"><h3>🎯 Target distribution</h3>', unsafe_allow_html=True)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# DATA TAB
# ════════════════════════════════════════════════════════════════════════════
elif tab == "data":

    st.markdown("### Load your dataset")

    data_source = st.radio(
        "Choose source",
        ["📊 Sample dataset", "📁 Upload CSV", "🔗 Paste URL"],
        label_visibility="collapsed"
    )

    if data_source == "📊 Sample dataset":
        n = st.slider("Number of records", 500, 5000, 1500, step=100)
        if st.button("Generate sample data"):
            st.session_state.df = generate_data(n)
            st.success(f"✅ Generated {n:,} records")
            st.rerun()
        st.markdown(f"""
        <div class="m-card">
          <h3>📋 Current dataset</h3>
          <div class="d-row"><span class="lbl">Rows</span><span class="val">{len(df):,}</span></div>
          <div class="d-row"><span class="lbl">Features</span><span class="val">{len(df.columns)-1}</span></div>
          <div class="d-row"><span class="lbl">Default rate</span><span class="val">{df['loan_status'].mean():.1%}</span></div>
          <div class="d-row"><span class="lbl">Missing values</span><span class="val">{df.isnull().sum().sum()}</span></div>
        </div>
        """, unsafe_allow_html=True)

    elif data_source == "📁 Upload CSV":
        st.markdown("""
        <div class="m-card">
          <h3>📤 Upload your CSV</h3>
          <div style="background:#f8f8f8;border:1.5px dashed #ccc;border-radius:12px;padding:24px;text-align:center;margin-bottom:10px">
            <div style="font-size:28px;color:#aaa;margin-bottom:8px">📄</div>
            <p style="font-size:13px;color:#666;margin:0 0 4px">Drop your CSV here or use the uploader below</p>
            <p style="font-size:11px;color:#aaa;margin:0">Supports .csv · Max 50MB</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Choose CSV file", type=["csv"], label_visibility="collapsed")
        if uploaded:
            try:
                new_df = pd.read_csv(uploaded)
                st.success(f"✅ Loaded {len(new_df):,} rows · {len(new_df.columns)} columns")
                st.dataframe(new_df.head(5), use_container_width=True)

                # Check required columns
                required = ["loan_amnt","int_rate","annual_inc","dti","loan_status"]
                missing_cols = [c for c in required if c not in new_df.columns]
                if missing_cols:
                    st.warning(f"⚠️ Missing columns: {', '.join(missing_cols)}\nUsing sample data for training.")
                else:
                    # Engineer features
                    new_df["loan_inc_ratio"] = new_df["loan_amnt"] / new_df["annual_inc"]
                    new_df["int_inc_ratio"]  = (new_df["int_rate"]/100)*new_df["loan_amnt"]/new_df["annual_inc"]
                    for col in ["emp_length","open_acc","revol_util","pub_rec","delinq_2yrs"]:
                        if col not in new_df.columns:
                            new_df[col] = 0
                    if st.button("Use this dataset"):
                        st.session_state.df = new_df
                        st.success("✅ Dataset updated! Go to Models tab to retrain.")
            except Exception as e:
                st.error(f"❌ Could not read file: {e}")

        st.markdown("""
        <div class="m-card">
          <h3>📋 Required CSV columns</h3>
          <div class="d-row"><span class="lbl">loan_amnt</span><span class="val">Loan amount ($)</span></div>
          <div class="d-row"><span class="lbl">int_rate</span><span class="val">Interest rate (%)</span></div>
          <div class="d-row"><span class="lbl">annual_inc</span><span class="val">Annual income ($)</span></div>
          <div class="d-row"><span class="lbl">dti</span><span class="val">Debt-to-income ratio</span></div>
          <div class="d-row"><span class="lbl">loan_status</span><span class="val">0 = No default, 1 = Default</span></div>
        </div>
        """, unsafe_allow_html=True)

    elif data_source == "🔗 Paste URL":
        url = st.text_input("CSV URL", placeholder="https://example.com/loans.csv")
        if st.button("Load from URL") and url:
            try:
                with st.spinner("Fetching data…"):
                    new_df = pd.read_csv(url)
                    st.session_state.df = new_df
                    st.success(f"✅ Loaded {len(new_df):,} rows from URL")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Could not load: {e}")

    # EDA section
    st.markdown("---")
    st.markdown("#### Explore data")
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "loan_status"]
    feat = st.selectbox("Feature distribution", num_cols)
    fig3 = px.histogram(df, x=feat, color="loan_status", barmode="overlay",
                        nbins=35, opacity=0.75,
                        color_discrete_map={0:"#4CAF50",1:"#F44336"})
    fig3.update_layout(height=220, margin=dict(t=10,b=10,l=0,r=0),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with st.expander("View raw data"):
        st.dataframe(df.head(20), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PREDICT TAB
# ════════════════════════════════════════════════════════════════════════════
elif tab == "predict":

    mode = st.radio("Input method", ["Manual input","Batch CSV","Test sample"],
                    horizontal=True, label_visibility="collapsed")

    if mode == "Manual input":
        with st.form("pred_form"):
            st.markdown("**Applicant details**")
            loan_amnt  = st.number_input("Loan amount ($)", 500, 40000, 10000, step=500)
            annual_inc = st.number_input("Annual income ($)", 10000, 300000, 60000, step=1000)
            int_rate   = st.slider("Interest rate (%)", 5.0, 30.0, 12.0, step=0.5)
            dti        = st.slider("Debt-to-income", 0.0, 40.0, 15.0, step=0.5)
            emp_length = st.slider("Employment (yrs)", 0, 10, 3)
            open_acc   = st.number_input("Open accounts", 1, 30, 8)
            revol_util = st.slider("Revolving utilisation (%)", 0.0, 100.0, 45.0)
            pub_rec    = st.selectbox("Public records", [0,1,2,3])
            delinq     = st.selectbox("Delinquencies (2yr)", [0,1,2,3,4])
            go = st.form_submit_button("🔍 Predict risk", use_container_width=True)

        if go:
            row = np.array([[loan_amnt, int_rate, annual_inc, dti, emp_length,
                             open_acc, revol_util, pub_rec, delinq,
                             loan_amnt/annual_inc,
                             (int_rate/100)*loan_amnt/annual_inc]])
            prob = res["clf"].predict_proba(row)[0][1]
            pct  = prob * 100
            if prob < 0.3:
                lvl, cls, icon, rec = "Low risk", "low", "✅", "Recommend approval"
            elif prob < 0.6:
                lvl, cls, icon, rec = "Medium risk", "mid", "⚠️", "Review required"
            else:
                lvl, cls, icon, rec = "High risk", "high", "🚨", "Recommend rejection"

            st.markdown(f"""
            <div class="risk-result {cls}">
              <div class="risk-pct {cls}">{pct:.1f}%</div>
              <div class="risk-text {cls}">{icon} {lvl} — {rec}</div>
            </div>
            <div class="m-card">
              <div class="d-row"><span class="lbl">Default probability</span><span class="val">{pct:.1f}%</span></div>
              <div class="d-row"><span class="lbl">Risk category</span>
                <span class="badge {'green' if cls=='low' else 'amber' if cls=='mid' else 'red'}">{lvl}</span>
              </div>
              <div class="d-row"><span class="lbl">Decision</span><span class="val">{rec}</span></div>
              <div class="d-row"><span class="lbl">Model</span><span class="val">{st.session_state.model_choice}</span></div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("_For portfolio demo purposes only — not financial advice._")

    elif mode == "Batch CSV":
        st.markdown("""
        <div class="m-card">
          <h3>📤 Batch prediction</h3>
          <p style="font-size:12px;color:#888;margin:0">Upload a CSV with applicant details to score all rows at once.</p>
        </div>
        """, unsafe_allow_html=True)
        batch_file = st.file_uploader("Upload batch CSV", type=["csv"], label_visibility="collapsed")
        if batch_file:
            batch_df = pd.read_csv(batch_file)
            st.success(f"Loaded {len(batch_df)} applicants")
            for col in FEATURES:
                if col not in batch_df.columns:
                    batch_df[col] = 0
            if "loan_inc_ratio" not in batch_df.columns:
                batch_df["loan_inc_ratio"] = batch_df["loan_amnt"] / batch_df["annual_inc"].replace(0,1)
            if "int_inc_ratio" not in batch_df.columns:
                batch_df["int_inc_ratio"] = (batch_df["int_rate"]/100)*batch_df["loan_amnt"]/batch_df["annual_inc"].replace(0,1)
            probs = res["clf"].predict_proba(batch_df[FEATURES].values)[:,1]
            batch_df["risk_score"] = (probs*100).round(1)
            batch_df["risk_level"] = pd.cut(probs, bins=[0,.3,.6,1],
                                             labels=["Low","Medium","High"])
            batch_df["decision"] = batch_df["risk_level"].map(
                {"Low":"Approve","Medium":"Review","High":"Reject"})
            st.dataframe(batch_df[["risk_score","risk_level","decision"]].head(20),
                         use_container_width=True)
            csv_out = batch_df.to_csv(index=False)
            st.download_button("⬇️ Download results CSV", csv_out,
                               file_name="predictions.csv", mime="text/csv")

    elif mode == "Test sample":
        idx = st.slider("Test sample index", 0, len(res["X_te"])-1, 0)
        sample = res["X_te"][idx:idx+1]
        prob   = res["clf"].predict_proba(sample)[0][1]
        actual = res["y_te"][idx]
        pct    = prob * 100
        cls    = "low" if prob < 0.3 else "mid" if prob < 0.6 else "high"
        st.markdown(f"""
        <div class="risk-result {cls}">
          <div class="risk-pct {cls}">{pct:.1f}%</div>
          <div class="risk-text {cls}">Predicted probability</div>
        </div>
        <div class="m-card">
          <div class="d-row"><span class="lbl">Actual outcome</span>
            <span class="badge {'green' if actual==0 else 'red'}">{'No default' if actual==0 else 'Default'}</span>
          </div>
          <div class="d-row"><span class="lbl">Correct?</span>
            <span class="val">{'✅ Yes' if (prob>0.5)==actual else '❌ No'}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if SHAP_AVAILABLE and res["shap_vals"] is not None:
            shap_df = pd.DataFrame({
                "Feature": FEATURES,
                "SHAP": res["shap_vals"][idx]
            }).sort_values("SHAP")
            fig_s = px.bar(shap_df, x="SHAP", y="Feature", orientation="h",
                           color="SHAP", color_continuous_scale="RdBu",
                           color_continuous_midpoint=0)
            fig_s.update_layout(height=280, margin=dict(t=10,b=10,l=0,r=0),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                coloraxis_showscale=False)
            st.markdown('<div class="m-card"><h3>🧠 SHAP explanation</h3>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_s, use_container_width=True,
                            config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MODELS TAB
# ════════════════════════════════════════════════════════════════════════════
elif tab == "models":

    model_opts = ["XGBoost","Random Forest","Logistic Regression"] if XGB_AVAILABLE \
                 else ["Random Forest","Logistic Regression"]
    choice = st.selectbox("Active model", model_opts,
                          index=model_opts.index(st.session_state.model_choice))
    if choice != st.session_state.model_choice:
        st.session_state.model_choice = choice
        st.rerun()

    # Metrics
    st.markdown(f"""
    <div class="m-card">
      <h3>📊 {choice} — performance</h3>
      <div class="d-row"><span class="lbl">Accuracy</span><span class="val">{res['acc']:.1%}</span></div>
      <div class="d-row"><span class="lbl">Precision</span><span class="val">{res['prec']:.1%}</span></div>
      <div class="d-row"><span class="lbl">Recall</span><span class="val">{res['rec']:.1%}</span></div>
      <div class="d-row"><span class="lbl">F1 score</span><span class="val">{res['f1']:.1%}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(res["y_te"], res["y_prob"])
    roc_auc = auc(fpr, tpr)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC = {roc_auc:.3f}",
                                  line=dict(color="#1976D2", width=2.5)))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random",
                                  line=dict(dash="dash", color="gray")))
    fig_roc.update_layout(height=240, margin=dict(t=10,b=10,l=0,r=0),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           showlegend=True, legend=dict(font_size=11),
                           xaxis_title="FPR", yaxis_title="TPR")
    st.markdown('<div class="m-card"><h3>📈 ROC curve</h3>', unsafe_allow_html=True)
    st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # Confusion matrix
    cm = confusion_matrix(res["y_te"], res["y_pred"])
    fig_cm = px.imshow(cm, text_auto=True,
                        labels=dict(x="Predicted", y="Actual"),
                        x=["No Default","Default"], y=["No Default","Default"],
                        color_continuous_scale="Blues")
    fig_cm.update_layout(height=240, margin=dict(t=10,b=10,l=0,r=0),
                          paper_bgcolor="rgba(0,0,0,0)")
    st.markdown('<div class="m-card"><h3>🎯 Confusion matrix</h3>', unsafe_allow_html=True)
    st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    if SHAP_AVAILABLE and res["shap_vals"] is not None:
        shap_df = pd.DataFrame({
            "Feature":    FEATURES,
            "Mean |SHAP|": np.abs(res["shap_vals"]).mean(axis=0)
        }).sort_values("Mean |SHAP|")
        fig_sh = px.bar(shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
                        color="Mean |SHAP|", color_continuous_scale="Oranges")
        fig_sh.update_layout(height=280, margin=dict(t=10,b=10,l=0,r=0),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              coloraxis_showscale=False)
        st.markdown('<div class="m-card"><h3>🧠 SHAP — global importance</h3>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_sh, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)