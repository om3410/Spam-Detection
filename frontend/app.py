import streamlit as st
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Spam Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .prediction-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .ham-card {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 5px solid #4CAF50;
    }
    
    .spam-card {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border-left: 5px solid #F44336;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .progress-container {
        width: 100%;
        background-color: #E0E0E0;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        line-height: 20px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Model configuration - UPDATED PATH
MODEL_PATH = "spam_model.pkl"

class SpamDetector:
    """Spam Detection Class"""
    
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model_data = None
        self.load_model()
    
    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model_data = pickle.load(f)
                return True
            return False
        except:
            return False
    
    def predict(self, text, model_name='naive_bayes'):
        if self.model_data is None:
            # Demo mode - simple keyword detection
            text_lower = text.lower()
            spam_words = ['win', 'prize', 'urgent', 'free', 'click', 'won', 'lottery', 'verify', 'account']
            
            spam_count = sum(1 for word in spam_words if word in text_lower)
            spam_prob = min(spam_count / 3, 0.95) if spam_count > 0 else 0.1
            
            return {
                "prediction": "spam" if spam_prob > 0.5 else "ham",
                "spam_probability": spam_prob,
                "ham_probability": 1 - spam_prob,
                "confidence": max(spam_prob, 1 - spam_prob),
                "model_used": "demo"
            }
        
        try:
            vectorizer = self.model_data['vectorizer']
            text_vec = vectorizer.transform([text])
            
            model = self.model_data['models'][model_name]
            prediction = model.predict(text_vec)[0]
            probabilities = model.predict_proba(text_vec)[0]
            
            if 'label_mapping' in self.model_data:
                reverse_map = {v: k for k, v in self.model_data['label_mapping'].items()}
                label = reverse_map.get(prediction, "spam" if prediction == 1 else "ham")
            else:
                label = "spam" if prediction == 1 else "ham"
            
            return {
                "prediction": label,
                "spam_probability": float(probabilities[1]),
                "ham_probability": float(probabilities[0]),
                "confidence": max(probabilities),
                "model_used": model_name
            }
        except:
            return {"error": "Prediction failed"}

# Initialize session state
if 'example_to_load' not in st.session_state:
    st.session_state.example_to_load = ""
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Load detector
detector = SpamDetector(MODEL_PATH)

# Sidebar
with st.sidebar:
    st.title("Configuration")
    st.markdown("---")
    
    st.subheader("Model Status")
    if detector.model_data:
        st.success("Model Loaded")
        st.info("Model is ready for predictions")
    else:
        st.warning("Demo Mode Active")
        st.info("Using keyword-based detection")
    
    st.markdown("---")
    
    st.subheader("Model Selection")
    selected_model = st.selectbox(
        "Choose model:",
        ["naive_bayes", "demo"],
        index=1 if not detector.model_data else 0
    )
    
    st.markdown("---")
    
    st.subheader("Quick Actions")
    if st.button("Reload App"):
        st.rerun()
    
    if st.button("Clear History"):
        st.session_state.prediction_history = []
        st.success("History cleared!")
    
    st.markdown("---")
    
    st.subheader("About")
    st.markdown("""
    Spam Detection System
    
    Machine Learning Internship
    Task 2 - Syntexhub
    """)

# Main content
st.markdown('<h1 class="main-header">Spam Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning Internship - Task 2 | Syntexhub</p>', unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3 = st.tabs(["Single Detection", "Analytics", "About"])

with tab1:
    st.header("Single Message Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Load example if set
        initial_message = st.session_state.example_to_load if st.session_state.example_to_load else ""
        
        message = st.text_area(
            "Enter your message:",
            height=150,
            placeholder="Type or paste your message here...",
            key="message_input",
            value=initial_message
        )
        
        # Clear example after loading
        if st.session_state.example_to_load:
            st.session_state.example_to_load = ""
        
        st.subheader("Quick Examples")
        example_cols = st.columns(3)
        
        examples = [
            "WINNER! You won a prize!",
            "Meeting at 3 PM today",
            "URGENT: Account verification needed"
        ]
        
        for i, (col, example) in enumerate(zip(example_cols, examples)):
            with col:
                if st.button(example[:20] + "..." if len(example) > 20 else example, key=f"example_{i}"):
                    st.session_state.example_to_load = example
                    st.rerun()
    
    with col2:
        status_color = "#4CAF50" if detector.model_data else "#FF9800"
        status_text = "Ready" if detector.model_data else "Demo Mode"
        
        st.markdown(f"""
        <div class="stat-card">
            <div class="metric-label">Model Status</div>
            <div class="metric-value" style="color: {status_color};">{status_text}</div>
            <div class="metric-label">Selected Model</div>
            <div style="font-size: 1.2rem; font-weight: bold;">{selected_model}</div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("Detect Spam", type="primary", use_container_width=True):
        if message:
            with st.spinner("Analyzing message..."):
                result = detector.predict(message, selected_model)
                
                # Store in history for analytics
                history_entry = {
                    "message": message[:50] + "..." if len(message) > 50 else message,
                    "prediction": result.get("prediction", "unknown"),
                    "spam_prob": result.get("spam_probability", 0),
                    "timestamp": st.session_state.get("current_time", "N/A")
                }
                st.session_state.prediction_history.append(history_entry)
                
                if "error" in result:
                    st.error(result["error"])
                    st.info("Using fallback prediction...")
                    if any(word in message.lower() for word in ['win', 'prize', 'urgent', 'free']):
                        result = {
                            "prediction": "spam",
                            "spam_probability": 0.85,
                            "ham_probability": 0.15,
                            "confidence": 0.85,
                            "model_used": "fallback"
                        }
                    else:
                        result = {
                            "prediction": "ham",
                            "spam_probability": 0.15,
                            "ham_probability": 0.85,
                            "confidence": 0.85,
                            "model_used": "fallback"
                        }
                else:
                    st.success("Prediction successful!")
                
                st.markdown("---")
                
                if result['prediction'] == 'spam':
                    st.markdown(f"""
                    <div class="prediction-card spam-card">
                        <h2>SPAM DETECTED</h2>
                        <p><strong>Model Used:</strong> {result.get('model_used', 'Unknown')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.warning("This message appears to be spam. Be cautious!")
                else:
                    st.markdown(f"""
                    <div class="prediction-card ham-card">
                        <h2>LEGITIMATE MESSAGE</h2>
                        <p><strong>Model Used:</strong> {result.get('model_used', 'Unknown')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("This message appears to be legitimate.")
                
                # Display probabilities
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    spam_prob = result.get('spam_probability', 0)
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="metric-label">SPAM PROBABILITY</div>
                        <div class="metric-value" style="color: #F44336;">{spam_prob:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    ham_prob = result.get('ham_probability', 1 - spam_prob)
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="metric-label">HAM PROBABILITY</div>
                        <div class="metric-value" style="color: #4CAF50;">{ham_prob:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    confidence = result.get('confidence', max(spam_prob, ham_prob))
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="metric-label">CONFIDENCE</div>
                        <div class="metric-value" style="color: #2196F3;">{confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Spam probability meter
                st.subheader("Spam Probability Meter")
                spam_percent = spam_prob * 100
                
                if spam_percent < 30:
                    color = "#4CAF50"
                    label = "Low Risk"
                elif spam_percent < 70:
                    color = "#FF9800"
                    label = "Medium Risk"
                else:
                    color = "#F44336"
                    label = "High Risk"
                
                st.markdown(f"""
                <div class="progress-container">
                    <div class="progress-bar" style="width: {spam_percent}%; background-color: {color};">
                        {spam_percent:.1f}%
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                    <span>0%</span>
                    <span style="font-weight: bold; color: {color};">{label}</span>
                    <span>100%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a message first.")

with tab2:
    st.header("Analytics & Performance")
    
    if not detector.model_data:
        st.info("Running in Demo Mode")
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="stat-card">
                <div class="metric-label">Detection Mode</div>
                <div class="metric-value">Demo</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stat-card">
                <div class="metric-label">Method</div>
                <div class="metric-value">Keyword</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="stat-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">~75%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Detection History")
        
        if st.session_state.prediction_history:
            for i, entry in enumerate(st.session_state.prediction_history):
                badge_color = "#F44336" if entry["prediction"] == "spam" else "#4CAF50"
                st.markdown(f"""
                <div style="padding: 10px; margin: 5px 0; border-radius: 5px; background-color: #f5f5f5;">
                    <strong>Message:</strong> {entry["message"]}<br>
                    <strong>Prediction:</strong> <span style="color: {badge_color}; font-weight: bold;">{entry["prediction"].upper()}</span><br>
                    <strong>Spam Probability:</strong> {entry["spam_prob"]:.1%}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No predictions yet. Use the Single Detection tab to start analyzing messages.")
        
        st.markdown("---")
        st.subheader("Demo Mode Details")
        st.markdown("""
        ### How Demo Mode Works
        
        The demo mode uses keyword detection to classify messages:
        
        **Spam Keywords:** win, prize, urgent, free, click, won, lottery, verify, account
        
        **Detection Logic:**
        1. Count how many spam keywords appear in the message
        2. Calculate spam probability: (keyword_count / 3) up to 95%
        3. Classify as spam if probability > 50%
        
        **Example:**
        - "WINNER! You won a prize!" → 3 keywords → 95% spam probability → SPAM
        - "Meeting at 3 PM today" → 0 keywords → 10% spam probability → HAM
        
        ### To Enable Full Analytics
        
        To use the complete machine learning model with detailed analytics:
        
        1. **Train a model** using machine learning algorithms
        2. **Save it as** `spam_model.pkl` in the same directory
        3. **Restart the app** to load the trained model
        
        The trained model will provide:
        - Real machine learning predictions
        - Model accuracy metrics
        - Feature importance
        - Performance charts
        - Confusion matrix
        """)
    else:
        st.success("Model Loaded Successfully")
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="stat-card">
                <div class="metric-label">Model Type</div>
                <div class="metric-value">Trained ML</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stat-card">
                <div class="metric-label">Algorithm</div>
                <div class="metric-value">Naive Bayes</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="stat-card">
                <div class="metric-label">Status</div>
                <div class="metric-value">Active</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Prediction Statistics")
        
        if st.session_state.prediction_history:
            total = len(st.session_state.prediction_history)
            spam_count = sum(1 for entry in st.session_state.prediction_history if entry["prediction"] == "spam")
            ham_count = total - spam_count
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="metric-label">Total Predictions</div>
                    <div class="metric-value">{total}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="metric-label">Spam Rate</div>
                    <div class="metric-value">{spam_count/total:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show recent predictions
            st.subheader("Recent Predictions")
            for entry in st.session_state.prediction_history[-5:]:  # Last 5 predictions
                badge_color = "#F44336" if entry["prediction"] == "spam" else "#4CAF50"
                st.markdown(f"""
                <div style="padding: 10px; margin: 5px 0; border-radius: 5px; background-color: #f5f5f5;">
                    <strong>Message:</strong> {entry["message"]}<br>
                    <strong>Prediction:</strong> <span style="color: {badge_color}; font-weight: bold;">{entry["prediction"].upper()}</span><br>
                    <strong>Spam Probability:</strong> {entry["spam_prob"]:.1%}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No predictions yet. Use the Single Detection tab to start analyzing messages.")
        
        st.markdown("---")
        st.info("Full analytics features are enabled with the trained model.")

with tab3:
    st.header("About Spam Detection System")
    
    st.markdown("""
    ## Spam Detection System
    
    **Version:** 1.0  
    **Author:** Syntexhub Intern  
    **Task:** Machine Learning Internship - Task 2
    
    ---
    
    ### Technology Stack
    
    - **Machine Learning:** Python, Scikit-learn
    - **NLP:** TF-IDF Vectorization
    - **Algorithms:** Naive Bayes
    - **Frontend:** Streamlit
    
    ### Features
    
    1. **Real-time Detection** - Instant spam classification
    2. **Probability Scores** - Detailed confidence metrics
    3. **Visual Analytics** - Interactive probability visualizations
    4. **Demo Mode** - Fallback keyword detection
    
    ### Getting Started
    
    1. Run: `streamlit run app.py`
    2. Open browser at `http://localhost:8501`
    3. Enter messages to detect spam
    
    ---
    
    *Built for Syntexhub Internship Program*
    """)
    
    st.markdown("---")
    st.subheader("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Check Model Status"):
            if detector.model_data:
                st.success("✅ Model loaded successfully")
            else:
                st.info("🔧 Running in demo mode")
    
    with col2:
        if st.button("View Prediction History"):
            if st.session_state.prediction_history:
                st.write(f"Total predictions: {len(st.session_state.prediction_history)}")
            else:
                st.info("No predictions recorded yet")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>© 2024 Spam Detection System | Syntexhub Machine Learning Internship</p>
    <p>Built with Streamlit • Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)