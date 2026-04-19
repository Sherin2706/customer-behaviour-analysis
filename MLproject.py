import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pyttsx3

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Shopping Analysis", layout="wide")

# ---------------------------
# 🎨 DARK UI
# ---------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #1e1b4b);
    color: white;
}

.title {
    text-align: center;
    font-size: 34px;
    font-weight: 600;
    color: #c084fc;
    margin-bottom: 20px;
}

.card {
    background: #1e293b;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(168,85,247,0.15);
    margin-bottom: 15px;
}

.stButton>button {
    background: linear-gradient(45deg, #6366f1, #a855f7);
    color: white;
    border-radius: 8px;
    height: 2.6em;
    font-weight: 500;
}

h2, h3 {
    color: #a78bfa;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🛒 Online Shopping Behaviour Analysis</div>", unsafe_allow_html=True)

# ---------------------------
# 🔊 VOICE
# ---------------------------
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ---------------------------
# LOAD MODEL
# ---------------------------
model = joblib.load("model.pkl")
accuracy = joblib.load("accuracy.pkl")
columns = joblib.load("columns.pkl")

# ---------------------------
# UPLOAD
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
file = st.file_uploader("📂 Upload Dataset", type=["csv"])
st.markdown("</div>", unsafe_allow_html=True)

if file is not None:
    data = pd.read_csv(file)

    # ---------------------------
    # PREVIEW
    # ---------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📄 Dataset Preview")
    st.dataframe(data.head())
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------
    # CLEAN DATA
    # ---------------------------
    if "Purchase_Decision" in data.columns:
        data["Purchase_Decision"] = (
            data["Purchase_Decision"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({
                "yes": 1, "y": 1, "1": 1,
                "no": 0, "n": 0, "0": 0
            })
        )

    data = data.drop(["Name","Timestamp"], axis=1, errors="ignore")

    data = data.drop(
        columns=[col for col in data.columns if "No_Purchase_Reason" in col],
        errors="ignore"
    )

    X = data.drop("Purchase_Decision", axis=1, errors="ignore")

    # ---------------------------
    # ENCODE + ALIGN
    # ---------------------------
    X = pd.get_dummies(X)
    X = X.reindex(columns=columns, fill_value=0)

    # ---------------------------
    # BUTTON
    # ---------------------------
    if st.button("🚀 Predict Overall Behaviour"):

        # ---------------------------
        # ACCURACY
        # ---------------------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Model Accuracy (From Colab)")
        st.success(f"Accuracy: {accuracy:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------
        # MODEL PREDICTION
        # ---------------------------
        preds = model.predict(X)

        buy_count = sum(preds)
        total = len(preds)

        buy_percent = (buy_count / total) * 100
        not_buy_percent = 100 - buy_percent

        # ---------------------------
        # RESULT TEXT
        # ---------------------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📢 Final Prediction")

        st.success(f"{buy_percent:.1f}% customers are likely to BUY")
        st.error(f"{not_buy_percent:.1f}% customers are NOT likely to BUY")

        # 🔊 Voice
        speak(f"{buy_percent:.0f} percent customers will buy and {not_buy_percent:.0f} percent customers will not buy")

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------
        # 🥧 PIE CHART (SMALL)
        # ---------------------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Customer Buying Behaviour")

        labels = ["Buy", "Not Buy"]
        sizes = [buy_percent, not_buy_percent]
        colors = ["#22c55e", "#ef4444"]

        fig, ax = plt.subplots(figsize=(2.2,2.2))

        ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            textprops={'fontsize':7},
            startangle=90
        )

        # Center align
        col1, col2, col3 = st.columns([1,1.5,1])
        with col2:
            st.pyplot(fig, use_container_width=False)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("<center>Developed using Machine Learning & Streamlit</center>", unsafe_allow_html=True)