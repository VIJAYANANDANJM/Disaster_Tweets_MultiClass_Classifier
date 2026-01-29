# dashboard/app.py
import streamlit as st
from datetime import datetime
from utils import classify_tweet

# Page setup
st.set_page_config(page_title="Disaster Tweet Dashboard", layout="wide")
st.title("🌍 Disaster Tweet Classifier Dashboard")

# Sidebar filters
classes = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation"
]
selected_class = st.sidebar.selectbox("Filter by class", ["All"] + classes)

# Input form
tweet = st.text_input("Enter a tweet:")
if st.button("Classify"):
    result = classify_tweet(tweet)
    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Store results in session state
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].insert(0, result)  # newest first

# Display results
if "history" in st.session_state:
    st.subheader("Incoming Tweets")
    filtered = st.session_state["history"]
    if selected_class != "All":
        filtered = [r for r in filtered if r["label"] == selected_class]

    for r in filtered:
        st.markdown(f"""
        **Tweet:** {r['text']}  
        **Class:** {r['label']}  
        **Confidence:** {r['confidence']:.2f}  
        **Timestamp:** {r['timestamp']}  
        ---
        """)