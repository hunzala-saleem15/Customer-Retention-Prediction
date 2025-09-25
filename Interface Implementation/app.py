from flask import Flask, render_template, request, redirect, url_for, session, make_response, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import os
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import pdfkit   
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html
from preprocess import preprocess_single_input  
import plotly.graph_objects as go
from plotly.io import to_html
from flask import make_response

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)

models = {
    "Logistic Regression": joblib.load("models/logistic_regression_model.pkl"),
    "Random Forest": joblib.load("models/random_forest_model.pkl"),
    "Stacking Classifier": joblib.load("models/stacking_ensemble_model.pkl")
}
model_intros = {
    "Logistic Regression": "A statistical model that estimates churn probability using a logistic function.",
    "Random Forest": "An ensemble method that combines multiple decision trees for better accuracy.",
    "XGBoost": "A gradient boosting algorithm known for its performance and speed."
}

managers = {
    "admin": generate_password_hash("admin123")
}


# -------------------------
# Helper function for predictions
# -------------------------
def get_predictions(X):
    results = {}
    label_map = {0: "Not Churn", 1: "Churn"}
    for name, model in models.items():
        if name == "Neural Network":
            prob = float(model.predict(X).flatten()[0]) * 100
            pred = 1 if prob > 50 else 0
        else:
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0][1] * 100

        results[name] = {
            "prediction": label_map[int(pred)],
            "probability": round(prob, 2)
        }
    return results

# -------------------------
# Data Understanding Page
# -------------------------
@app.route("/data_overview")
def data_overview():
    if "user" not in session:
        return redirect(url_for("login"))

    # Load dataset
    df = pd.read_csv("data/churn_data.csv")

    # Dataset info
    total_rows, total_cols = df.shape
    columns = df.columns.tolist()

    # Sample data (first 10 rows)
    sample_data = df.head(10).to_dict(orient="records")

    # Numeric stats
    numeric_cols = df.select_dtypes(include='number').columns
    numeric_stats = {}
    for col in numeric_cols:
        numeric_stats[col] = {
            "mean": round(df[col].mean(),2),
            "median": round(df[col].median(),2),
            "min": round(df[col].min(),2),
            "max": round(df[col].max(),2)
        }

    # Categorical top values
    categorical_cols = df.select_dtypes(exclude='number').columns
    categorical_top = {}
    for col in categorical_cols:
        categorical_top[col] = df[col].mode()[0]

    return render_template(
        "data_overview.html",
        total_rows=total_rows,
        total_cols=total_cols,
        columns=columns,
        sample_data=sample_data,
        numeric_stats=numeric_stats,
        categorical_top=categorical_top
    )

# -------------------------
# Routes
# -------------------------
@app.route('/')
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("home.html", home=True)

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username in managers and check_password_hash(managers[username], password):
            session["user"] = username
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route('/logout')
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# -------------------------
# Model Introductions
# -------------------------
model_intros = {
    "Logistic Regression": "Logistic Regression is a simple yet powerful statistical model that estimates churn probability. It is highly interpretable and serves as a strong baseline.",
    "Random Forest": "Random Forest is an ensemble method that combines multiple decision trees to deliver robust and accurate predictions while reducing overfitting.",
    "Stacking Classifier": "Stacking Classifier is an advanced ensemble model that blends different algorithms and leverages their strengths for more reliable predictions."
}

# -------------------------
# Prediction Page
# -------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        raw_input = request.form.to_dict()
        X = preprocess_single_input(raw_input)
        results = get_predictions(X)

        # Top-3 enhanced gauge charts
        top_results = dict(list(results.items())[:3])
        top_charts = []

        for model_name, res in top_results.items():
            prob = float(res["probability"])
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob,
                delta={'reference': 50, 'increasing': {'color': 'red'}},
                title={'text': f"{model_name} Churn Probability", 'font': {'size': 18, 'family': 'Arial'}},
                number={'suffix': '%', 'font': {'size': 22, 'family': 'Arial', 'color': '#0d214d'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#0d214d"},
                    'bar': {'color': "#42a5f5", 'thickness': 0.4, 'line': {'color': '#42a5f5', 'width': 0.5}},
                    'bgcolor': "rgba(255,255,255,0.08)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(255,255,255,0.3)",
                    'steps': [
                        {'range': [0, 10], 'color': '#BBDEFB'},
                        {'range': [10, 20], 'color': '#90CAF9'},
                        {'range': [20, 30], 'color': '#64B5F6'},
                        {'range': [30, 40], 'color': '#42A5F5'},
                        {'range': [40, 50], 'color': '#2196F3'},
                        {'range': [50, 60], 'color': '#1E88E5'},
                        {'range': [60, 70], 'color': '#1976D2'},
                        {'range': [70, 80], 'color': '#1565C0'},
                        {'range': [80, 90], 'color': '#0D47A1'},
                        {'range': [90,100], 'color': '#0B3D91'},
                    ],
                    'threshold': {
                        'line': {'color': "#FF1744", 'width': 4},
                        'thickness': 0.85,
                        'value': prob
                    }
                }
            ))

            fig.update_layout(
                paper_bgcolor="rgba(255,255,255,0.05)",
                font={'color': "#0d214d", 'family': "Arial"},
                margin={'t': 50, 'b': 50, 'l': 50, 'r': 50},
                transition={'duration': 1200, 'easing': 'cubic-in-out'}
            )
            top_charts.append(to_html(fig, full_html=False, include_plotlyjs='cdn'))

        top_results_charts = list(zip(top_results.items(), top_charts))
        return render_template(
            "results.html",
            results=results,
            top_results_charts=top_results_charts,
            model_intros=model_intros,
            raw_input = raw_input
        )

    return render_template("predict.html")


@app.route('/results', methods=["POST"])
def results():
    if "user" not in session:
        return redirect(url_for("login"))

    raw_input = request.form.to_dict()
    X = preprocess_single_input(raw_input)
    results = get_predictions(X)

    # Top-3 enhanced gauges
    top_results = dict(list(results.items())[:3])
    top_charts = []

    for model_name, res in top_results.items():
        prob = float(res["probability"])
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob,
            delta={'reference': 50, 'increasing': {'color': 'red'}},
            title={'text': f"{model_name} Churn Probability", 'font': {'size': 18, 'family': 'Arial'}},
            number={'suffix': '%', 'font': {'size': 22, 'family': 'Arial', 'color': '#0d214d'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#0d214d"},
                'bar': {'color': "#42a5f5", 'thickness': 0.4, 'line': {'color': '#42a5f5', 'width': 0.5}},
                'bgcolor': "rgba(255,255,255,0.08)",
                'borderwidth': 2,
                'bordercolor': "rgba(255,255,255,0.3)",
                'steps': [
                    {'range': [0, 10], 'color': '#BBDEFB'},
                    {'range': [10, 20], 'color': '#90CAF9'},
                    {'range': [20, 30], 'color': '#64B5F6'},
                    {'range': [30, 40], 'color': '#42A5F5'},
                    {'range': [40, 50], 'color': '#2196F3'},
                    {'range': [50, 60], 'color': '#1E88E5'},
                    {'range': [60, 70], 'color': '#1976D2'},
                    {'range': [70, 80], 'color': '#1565C0'},
                    {'range': [80, 90], 'color': '#0D47A1'},
                    {'range': [90,100], 'color': '#0B3D91'},
                ],
                'threshold': {
                    'line': {'color': "#FF1744", 'width': 4},
                    'thickness': 0.85,
                    'value': prob
                }
            }
        ))

        fig.update_layout(
            paper_bgcolor="rgba(255,255,255,0.05)",
            font={'color': "#0d214d", 'family': "Arial"},
            margin={'t': 50, 'b': 50, 'l': 50, 'r': 50},
            transition={'duration': 1200, 'easing': 'cubic-in-out'}
        )
        top_charts.append(to_html(fig, full_html=False, include_plotlyjs='cdn'))

    top_results_charts = list(zip(top_results.items(), top_charts))
    return render_template(
        "results.html",
        results=results,
        top_results_charts=top_results_charts,
        model_intros=model_intros
    )
#--------------------------
# Report & Dashboard
#--------------------------
@app.route('/report', methods=["GET", "POST"])
def report():
    if "user" not in session:
        return redirect(url_for("login"))

    df = pd.read_csv("data/churn_data.csv")
    
    if request.method == "POST" and request.form:
        raw_input = request.form.to_dict()
        X = preprocess_single_input(raw_input)
        prediction_results = get_predictions(X)
    else:
        # GET request: redirect back to predict form
        return redirect(url_for("predict"))

    # Extract UI input values
    ui_gender   = (raw_input.get("gender") or "").strip()
    ui_senior   = (raw_input.get("senior_citizen") or "").strip()
    ui_partner  = (raw_input.get("partner") or "").strip()
    ui_depend   = (raw_input.get("dependents") or "").strip()
    ui_contract = (raw_input.get("contract") or "").strip()
    ui_payment  = (raw_input.get("payment_method") or "").strip()
    ui_internet = (raw_input.get("internet_service") or "").strip()
    ui_monthly  = float(raw_input.get("monthly_charges", 0) or 0)
    ui_tenure   = float(raw_input.get("tenure_months", 0) or 0)
    ui_total    = float(raw_input.get("total_charges", 0) or 0)

    # Ensure necessary columns exist in df
    for col in ["Contract","Payment Method","Internet Service","Senior Citizen",
                "Partner","Dependents","Gender","Monthly Charges",
                "Tenure Months","Total Charges"]:
        if col not in df.columns:
            df[col] = 0 if "Charges" in col or "Months" in col else "Unknown"

    graphs_html = []

    # ---------- 1) Monthly Charges Distribution ----------
    if ui_monthly > 0:
        fig1 = px.histogram(df, x="Monthly Charges", color="Churn Value", nbins=30,
                            barmode="overlay", opacity=0.75,
                            title="Monthly Charges Distribution by Churn")
        fig1.add_vline(x=ui_monthly, line_dash="dash", line_color="red",
                       annotation_text=f"User: {ui_monthly:.2f}", annotation_position="top")
        graphs_html.append(to_html(fig1, full_html=False, include_plotlyjs='cdn'))

    # ---------- 2) Tenure Distribution ----------
    if ui_tenure > 0:
        fig2 = px.histogram(df, x="Tenure Months", color="Churn Value", nbins=30,
                            barmode="stack", title="Tenure Distribution by Churn")
        fig2.add_vline(x=ui_tenure, line_dash="dash", line_color="red",
                       annotation_text=f"User: {ui_tenure:.0f} mo", annotation_position="top")
        graphs_html.append(to_html(fig2, full_html=False, include_plotlyjs=False))

    # ---------- 3) Contract vs Churn Rate ----------
    if ui_contract:
        cr_contract = df.groupby("Contract")["Churn Value"].mean().reset_index()
        colors = ["#87CEFA" if str(x).lower()!=ui_contract.lower() else "#EF5350" for x in cr_contract["Contract"]]
        fig3 = go.Figure(go.Bar(
            x=cr_contract["Contract"], y=(cr_contract["Churn Value"]*100.0),
            marker_color=colors,
            hovertemplate="Contract=%{x}<br>Churn Rate=%{y:.1f}%<extra></extra>"
        ))
        fig3.update_layout(title="Contract Type vs Churn Rate",
                           xaxis_title="Contract Type", yaxis_title="Churn Rate (%)")
        graphs_html.append(to_html(fig3, full_html=False, include_plotlyjs=False))

    # ---------- 4) Payment Method vs Churn Rate ----------
    if ui_payment:
        cr_pay = df.groupby("Payment Method")["Churn Value"].mean().reset_index()
        colors = ["#90CAF9" if str(x).lower()!=ui_payment.lower() else "#EF5350" for x in cr_pay["Payment Method"]]
        fig4 = go.Figure(go.Bar(
            x=cr_pay["Payment Method"], y=(cr_pay["Churn Value"]*100.0),
            marker_color=colors,
            hovertemplate="Payment=%{x}<br>Churn Rate=%{y:.1f}%<extra></extra>"
        ))
        fig4.update_layout(title="Payment Method vs Churn Rate",
                           xaxis_title="Payment Method", yaxis_title="Churn Rate (%)")
        graphs_html.append(to_html(fig4, full_html=False, include_plotlyjs=False))

    # ---------- 5) Internet Service vs Churn Rate ----------
    if ui_internet:
        cr_int = df.groupby("Internet Service")["Churn Value"].mean().reset_index()
        colors = ["#A5D6A7" if str(x).lower()!=ui_internet.lower() else "#EF5350" for x in cr_int["Internet Service"]]
        fig5 = go.Figure(go.Bar(
            x=cr_int["Internet Service"], y=(cr_int["Churn Value"]*100.0),
            marker_color=colors,
            hovertemplate="Internet=%{x}<br>Churn Rate=%{y:.1f}%<extra></extra>"
        ))
        fig5.update_layout(title="Internet Service vs Churn Rate",
                           xaxis_title="Internet Service", yaxis_title="Churn Rate (%)")
        graphs_html.append(to_html(fig5, full_html=False, include_plotlyjs=False))

    # ---------- 6) Similar Customers Cohort Gauge ----------
    if ui_contract and ui_internet and ui_senior:
        df["_SeniorBool"] = df["Senior Citizen"].astype(str).str.lower().isin(["yes","1","true"])
        ui_senior_bool = str(ui_senior).lower() in ["yes","1","true"]
        cohort = df[
            (df["Contract"].astype(str).str.lower()==ui_contract.lower()) &
            (df["Internet Service"].astype(str).str.lower()==ui_internet.lower()) &
            (df["_SeniorBool"]==ui_senior_bool)
        ]
        cohort_rate = float(cohort["Churn Value"].mean()) if len(cohort)>0 else float(df["Churn Value"].mean())
        cohort_pct  = round(cohort_rate*100.0, 1)
        overall_pct = round(float(df["Churn Value"].mean()*100.0), 1)
        fig6 = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=cohort_pct, number={"suffix":"%"},
            delta={"reference": overall_pct, "suffix":"%"},
            gauge={"axis":{"range":[0,100]}, "bar":{"color":"#EF5350"}},
            title={"text":"Churn Risk: Similar Customers"}
        ))
        graphs_html.append(to_html(fig6, full_html=False, include_plotlyjs=False))

    # ---------- 7) Total Charges Distribution ----------
    if ui_total > 0:
        fig7 = px.histogram(df, x="Total Charges", color="Churn Value", nbins=30,
                            barmode="overlay", opacity=0.75,
                            title="Total Charges Distribution by Churn")
        fig7.add_vline(x=ui_total, line_dash="dash", line_color="red",
                       annotation_text=f"User: {ui_total:.2f}", annotation_position="top")
        graphs_html.append(to_html(fig7, full_html=False, include_plotlyjs=False))

    # ---------- 8) Scatterplot: Tenure vs Monthly Charges ----------
    if ui_tenure > 0 and ui_monthly > 0:
        fig8 = px.scatter(df, x="Tenure Months", y="Monthly Charges", color="Churn Value",
                          opacity=0.6, title="Tenure vs Monthly Charges (Churn Split)")
        fig8.add_trace(go.Scatter(
            x=[ui_tenure], y=[ui_monthly],
            mode="markers+text", marker=dict(color="red", size=12, symbol="x"),
            text=["You"], textposition="top center", name="User"
        ))
        graphs_html.append(to_html(fig8, full_html=False, include_plotlyjs=False))

    # ---------- 9) Heatmap: Contract × Payment Method ----------
    if ui_contract and ui_payment:
        heat_df = df.groupby(["Contract","Payment Method"])["Churn Value"].mean().reset_index()
        heat_df["ChurnPct"] = heat_df["Churn Value"]*100
        fig9 = px.density_heatmap(heat_df, x="Contract", y="Payment Method", z="ChurnPct",
                                  color_continuous_scale="Reds", title="Churn % by Contract × Payment Method")
        fig9.add_trace(go.Scatter(
            x=[ui_contract], y=[ui_payment], mode="markers+text",
            marker=dict(color="blue", size=15, symbol="star"), text=["You"], name="User"
        ))
        graphs_html.append(to_html(fig9, full_html=False, include_plotlyjs=False))

    # ---------- 10) Prediction Gauge for All Models ----------
    if prediction_results:
        for model_name, res in prediction_results.items():
            prob = float(res["probability"])
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                title={"text": f"Predicted Churn Probability ({model_name})"},
                gauge={
                    "axis":{"range":[0,100]},
                    "bar":{"color":"#EF5350"},
                    "steps":[
                        {"range":[0,30], "color":"#C8E6C9"},
                        {"range":[30,60], "color":"#FFF59D"},
                        {"range":[60,100], "color":"#FFCDD2"},
                    ],
                }
            ))
            graphs_html.append(to_html(fig, full_html=False, include_plotlyjs=False))

    return render_template(
        "report.html",
        results=prediction_results,
        raw_input=raw_input,
        graphs_html=graphs_html
    )


@app.route("/comparison")
def model_comparison():
    if "user" not in session:
        return redirect(url_for("login"))

    # Example dummy comparison data
    comparison_data = [
        {"model": "Logistic Regression", "accuracy": "82%", "precision": "80%", "recall": "78%"},
        {"model": "Random Forest", "accuracy": "88%", "precision": "85%", "recall": "84%"},
        {"model": "XGBoost", "accuracy": "90%", "precision": "87%", "recall": "86%"}
    ]

    return render_template("comparison.html", comparison_data=comparison_data)

@app.route('/suggestions')
def suggestions():
    # --- Top Features Driving Churn (Random Forest / Feature Importance) ---
    top_features = [
        {
            'feature': 'Tenure',
            'importance': 0.19,
            'insight': 'New customers are at highest risk; churn decreases with longer tenure.',
            'action': 'Create an onboarding program for first 60 days with check-ins and small loyalty bonuses.'
        },
        {
            'feature': 'Contract Type (Month-to-month)',
            'importance': 0.17,
            'insight': 'Short-term contracts have higher churn.',
            'action': 'Offer incentives to switch to 1- or 2-year contracts.'
        },
        {
            'feature': 'Total Charges',
            'importance': 0.11,
            'insight': 'Lower-spending customers churn more frequently.',
            'action': 'Implement campaigns to increase engagement and perceived value.'
        },
        {
            'feature': 'Monthly Charges',
            'importance': 0.10,
            'insight': 'High monthly fees increase churn risk.',
            'action': 'Provide personalized value reviews to high-paying customers.'
        },
        {
            'feature': 'Online Security',
            'importance': 0.08,
            'insight': 'Lack of online security drives churn.',
            'action': 'Bundle security or run promotions highlighting its importance.'
        },
        {
            'feature': 'Tech Support',
            'importance': 0.07,
            'insight': 'Insufficient tech support leads to churn.',
            'action': 'Offer free trial support and improve accessibility.'
        },
        {
            'feature': 'Payment Method (Electronic Check)',
            'importance': 0.06,
            'insight': 'Manual payments correlate with higher churn.',
            'action': 'Encourage automatic payments with discounts or bonuses.'
        },
        {
            'feature': 'Internet Service (Fiber Optic)',
            'importance': 0.05,
            'insight': 'Fiber optic users churn more, possibly due to higher expectations or service issues.',
            'action': 'Conduct satisfaction surveys and review pricing/service reliability.'
        }
    ]

    # --- Contract Type Churn (Pie Chart) ---
    contract_churn = [
        {'contract': 'Month-to-month', 'churn_count': 1400},
        {'contract': 'One year', 'churn_count': 300},
        {'contract': 'Two year', 'churn_count': 169}
    ]

    # --- Churn vs Tenure (Line Graph) ---
    churn_vs_tenure = [
        {'month': 1, 'churn_rate': 0.25},
        {'month': 2, 'churn_rate': 0.22},
        {'month': 3, 'churn_rate': 0.20},
        {'month': 4, 'churn_rate': 0.15},
        {'month': 5, 'churn_rate': 0.13},
        {'month': 6, 'churn_rate': 0.12},
        {'month': 7, 'churn_rate': 0.11},
        {'month': 8, 'churn_rate': 0.10},
        {'month': 9, 'churn_rate': 0.09},
        {'month': 10, 'churn_rate': 0.08},
        {'month': 11, 'churn_rate': 0.07},
        {'month': 12, 'churn_rate': 0.06},
    ]

    return render_template(
        'suggestions.html',
        top_features=top_features,
        contract_churn=contract_churn,
        churn_vs_tenure=churn_vs_tenure
    )

# -------------------------
# Dashboard (updated with more graphs)
# -------------------------
@app.route('/dashboard')
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    df = pd.read_csv("data/churn_data.csv")
    if "Churn Value" in df.columns:
        df.rename(columns={"Churn Value": "Churn"}, inplace=True)

    total_customers = len(df)
    churned = df[df["Churn"] == 1].shape[0]
    not_churned = df[df["Churn"] == 0].shape[0]
    churn_rate = round((churned / total_customers) * 100, 2)

    # ---------- Chart 1: Churn Distribution ----------
    fig1 = px.histogram(df, x="Churn", color=df["Churn"].map({0:"Not Churn",1:"Churn"}),
                        barmode="overlay", title="Churn Distribution")
    chart1 = to_html(fig1, full_html=False, include_plotlyjs='cdn')

    # ---------- Chart 2: Contract vs Churn Rate ----------
    cr_contract = df.groupby("Contract")["Churn"].mean().reset_index()
    fig2 = px.bar(cr_contract, x="Contract", y="Churn", title="Contract Type vs Churn Rate",
                  labels={"Churn":"Churn Rate"})
    chart2 = to_html(fig2, full_html=False, include_plotlyjs=False)

    # ---------- Chart 3: Payment Method vs Churn Rate ----------
    cr_payment = df.groupby("Payment Method")["Churn"].mean().reset_index()
    fig3 = px.bar(cr_payment, x="Payment Method", y="Churn", title="Payment Method vs Churn Rate")
    chart3 = to_html(fig3, full_html=False, include_plotlyjs=False)

    # ---------- Chart 4: Internet Service vs Churn Rate ----------
    cr_internet = df.groupby("Internet Service")["Churn"].mean().reset_index()
    fig4 = px.bar(cr_internet, x="Internet Service", y="Churn", title="Internet Service vs Churn Rate")
    chart4 = to_html(fig4, full_html=False, include_plotlyjs=False)

    # ---------- Chart 5: Monthly Charges Distribution ----------
    fig5 = px.histogram(df, x="Monthly Charges", color=df["Churn"].map({0:"Not Churn",1:"Churn"}),
                        barmode="overlay", nbins=30, title="Monthly Charges Distribution")
    chart5 = to_html(fig5, full_html=False, include_plotlyjs=False)

    # ---------- Chart 6: Gender vs Churn ----------
    cr_gender = df.groupby("Gender")["Churn"].mean().reset_index()
    fig6 = px.bar(cr_gender, x="Gender", y="Churn", title="Gender vs Churn Rate")
    chart6 = to_html(fig6, full_html=False, include_plotlyjs=False)

    # ---------- Chart 7: Tenure Distribution ----------
    fig7 = px.histogram(df, x="Tenure Months", color=df["Churn"].map({0:"Not Churn",1:"Churn"}), nbins=30,
                        barmode="overlay", title="Tenure Distribution by Churn")
    chart7 = to_html(fig7, full_html=False, include_plotlyjs=False)

    # ---------- Chart 8: Internet Service vs Monthly Charges ----------
    fig8 = px.box(df, x="Internet Service", y="Monthly Charges", color=df["Churn"].map({0:"Not Churn",1:"Churn"}),
                  title="Internet Service vs Monthly Charges by Churn")
    chart8 = to_html(fig8, full_html=False, include_plotlyjs=False)

    # ---------- Chart 9: Payment Method vs Monthly Charges ----------
    fig9 = px.box(df, x="Payment Method", y="Monthly Charges", color=df["Churn"].map({0:"Not Churn",1:"Churn"}),
                  title="Payment Method vs Monthly Charges by Churn")
    chart9 = to_html(fig9, full_html=False, include_plotlyjs=False)

    # ---------- Chart 10: Tenure vs Monthly Charges Scatter ----------
    fig10 = px.scatter(df, x="Tenure Months", y="Monthly Charges", color=df["Churn"].map({0:"Not Churn",1:"Churn"}),
                       title="Tenure vs Monthly Charges Colored by Churn",
                       labels={"Tenure Months":"Tenure (Months)", "Monthly Charges":"Monthly Charges ($)"})
    chart10 = to_html(fig10, full_html=False, include_plotlyjs=False)

    return render_template("dashboard.html",
        total_customers=total_customers,
        churned=churned,
        not_churned=not_churned,
        churn_rate=churn_rate,
        chart1=chart1,
        chart2=chart2,
        chart3=chart3,
        chart4=chart4,
        chart5=chart5,
        chart6=chart6,
        chart7=chart7,
        chart8=chart8,
        chart9=chart9,
        chart10=chart10
    )

if __name__ == "__main__":
    app.run(debug=True)