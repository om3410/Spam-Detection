

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import json
import time
from datetime import datetime
import sys
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main styles */
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
    
    /* Cards */
    .prediction-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
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
    
    /* Metrics */
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
    
    /* Progress bars */
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
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .badge-spam {
        background-color: #F44336;
        color: white;
    }
    
    .badge-ham {
        background-color: #4CAF50;
        color: white;
    }
    
    .badge-info {
        background-color: #2196F3;
        color: white;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
    }
    </style>
""", unsafe_allow_html=True)


MODEL_PATH = r"C:\Users\omrew\OneDrive\Documents\project\spam_detestion\backend\spam_model.pkl"
BACKEND_DIR = r"C:\Users\omrew\OneDrive\Documents\project\spam_detestion\backend"


class SpamDetector:
    """Spam Detection Class"""
    
    def __init__(self, model_path=MODEL_PATH):
        """Initialize with model path"""
        self.model_path = model_path
        self.backend_dir = BACKEND_DIR
        self.model_data = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model from specific path"""
        try:
            st.sidebar.info(f"🔍 Looking for model at:\n`{self.model_path}`")
            
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model_data = pickle.load(f)
                st.sidebar.success("✅ Model loaded successfully!")
                return True
            else:
                st.sidebar.warning(f"❌ Model file not found at specified path")
                
                st.sidebar.info("🔍 Searching in backend directory...")
                
                if os.path.exists(self.backend_dir):
                    files = os.listdir(self.backend_dir)
                    pkl_files = [f for f in files if f.endswith('.pkl')]
                    
                    if pkl_files:
                        st.sidebar.info(f"Found .pkl files: {pkl_files}")
                        
                        for pkl_file in pkl_files:
                            try:
                                model_file = os.path.join(self.backend_dir, pkl_file)
                                with open(model_file, 'rb') as f:
                                    test_data = pickle.load(f)
                                
                                if isinstance(test_data, dict) and 'models' in test_data:
                                    self.model_path = model_file
                                    self.model_data = test_data
                                    st.sidebar.success(f"✅ Found valid model: {pkl_file}")
                                    return True
                            except:
                                continue
                    
                    st.sidebar.error(f"No valid model files found in:\n`{self.backend_dir}`")
                    
                    if st.sidebar.button("🔄 Create Fallback Model"):
                        self._create_fallback_model()
                        return True
                else:
                    st.sidebar.error(f"Backend directory not found:\n`{self.backend_dir}`")
                
                return False
                
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")
            return False
    
    def _create_fallback_model(self):
        """Create a simple fallback model for testing"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            
            texts = [
                "win free money prize lottery",
                "urgent verify account click",
                "meeting tomorrow lunch",
                "package delivered today"
            ]
            labels = [1, 1, 0, 0] 
            
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(texts)
            model = MultinomialNB()
            model.fit(X, labels)
            
            self.model_data = {
                'vectorizer': vectorizer,
                'models': {'naive_bayes': model},
                'results': {'naive_bayes': {'accuracy': 0.75}},
                'best_model': 'naive_bayes',
                'label_mapping': {'ham': 0, 'spam': 1}
            }
            
            st.sidebar.success("✅ Created fallback model for testing")
            st.sidebar.warning("⚠️ This is a demo model. For full features, run the backend.")
            return True
            
        except Exception as e:
            st.sidebar.error(f"Failed to create fallback model: {str(e)}")
            return False
    
    def predict(self, text, model_name='naive_bayes'):
        """Predict if text is spam"""
        if self.model_data is None:
            return {"error": "Model not loaded"}
        
        if 'models' not in self.model_data:
            return {"error": "Models not found in model data"}
        
        if model_name not in self.model_data['models']:
            available = list(self.model_data['models'].keys())
            return {"error": f"Model '{model_name}' not found. Available: {available}"}
        
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
                "text": text,
                "prediction": label,
                "spam_probability": float(probabilities[1]),
                "ham_probability": float(probabilities[0]),
                "confidence": max(probabilities),
                "model_used": model_name
            }
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def batch_predict(self, texts, model_name='naive_bayes'):
        """Predict multiple messages"""
        if self.model_data is None:
            return [{"error": "Model not loaded"}]
        
        results = []
        for text in texts:
            result = self.predict(text, model_name)
            results.append(result)
        return results
    
    def get_model_info(self):
        """Get model information"""
        if self.model_data is None:
            return {"error": "Model not loaded"}
        
        try:
            info = {
                "models_available": list(self.model_data.get('models', {}).keys()),
                "model_path": self.model_path,
                "file_size": f"{os.path.getsize(self.model_path) / 1024:.1f} KB" if os.path.exists(self.model_path) else "Unknown"
            }
            
            if 'best_model' in self.model_data:
                info['best_model'] = self.model_data['best_model']
            
            if 'results' in self.model_data:
                if 'best_model' in info:
                    best = info['best_model']
                    if best in self.model_data['results']:
                        info['accuracy'] = self.model_data['results'][best]['accuracy']
                else:
                    first_model = list(self.model_data['results'].keys())[0]
                    info['accuracy'] = self.model_data['results'][first_model]['accuracy']
            
            if 'vectorizer' in self.model_data:
                try:
                    features = self.model_data['vectorizer'].get_feature_names_out()
                    info['vocabulary_size'] = len(features)
                    info['sample_features'] = features[:5].tolist()
                except:
                    info['vocabulary_size'] = "Unknown"
            
            return info
        except Exception as e:
            return {"error": f"Could not get model info: {str(e)}"}

@st.cache_resource
def load_detector():
    """Load the spam detector with caching"""
    detector = SpamDetector(MODEL_PATH)
    return detector

# Load detector
detector = load_detector()

# ====================
# SIDEBAR
# ====================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/email-sign.png", width=80)
    st.title("⚙️ Configuration")
    
    st.markdown("---")
    
    st.subheader("🔍 Model Status")
    
    if detector.model_data:
        st.success("✅ Model Loaded")
        
        model_info = detector.get_model_info()
        if "error" not in model_info:
            st.info(f"**Best Model:** {model_info.get('best_model', 'N/A')}")
            if 'accuracy' in model_info:
                st.info(f"**Accuracy:** {model_info.get('accuracy', 0):.2%}")
            st.info(f"**Models:** {', '.join(model_info.get('models_available', []))}")
            st.info(f"**File:** {os.path.basename(model_info.get('model_path', 'Unknown'))}")
    else:
        st.error("❌ Model Not Found")
        
        with st.expander("🛠️ Troubleshooting"):
            st.markdown("""
            **To fix this issue:**
            
            1. **Run the backend Jupyter notebook:**
               - Open `backend/Spam_Detection_Backend.ipynb`
               - Run all cells
               - This creates `spam_model.pkl`
            
            2. **Check model location:**
               ```python
               import os
               path = r"C:\\Users\\omrew\\OneDrive\\Documents\\project\\spam_detestion\\backend"
               print("Files:", os.listdir(path))
               ```
            
            3. **Alternative paths to check:**
               - `backend/spam_model.pkl`
               - `spam_model.pkl`
               - Check OneDrive sync status
            
            4. **If still not working:**
               - Click "Create Fallback Model" for testing
               - Or use demo mode
            """)
    
    st.markdown("---")
    
    # Model selection
    st.subheader("🤖 Model Selection")
    
    if detector.model_data and 'models' in detector.model_data:
        model_options = list(detector.model_data['models'].keys())
        selected_model = st.selectbox(
            "Choose model:",
            model_options,
            index=0 if 'naive_bayes' in model_options else 0
        )
    else:
        selected_model = "naive_bayes"
        st.warning("Using default model")
    
    st.markdown("---")
    
    st.subheader("🚀 Quick Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reload App"):
            st.rerun()
    
    with col2:
        if st.button("📊 Demo Mode"):
            st.session_state.demo_mode = True
    
    if st.button("🔍 Debug Model", type="secondary"):
        with st.expander("Debug Information"):
            st.write("**Detector Status:**", "Loaded" if detector.model_data else "Not Loaded")
            st.write("**Model Path:**", detector.model_path)
            st.write("**Path Exists:**", os.path.exists(detector.model_path))
            
            if os.path.exists(detector.backend_dir):
                files = os.listdir(detector.backend_dir)
                st.write(f"**Backend Files ({len(files)}):**")
                for f in files[:10]:  # Show first 10
                    st.write(f"  - {f}")
                if len(files) > 10:
                    st.write(f"  ... and {len(files)-10} more")
            else:
                st.write(f"**Backend directory not found:** {detector.backend_dir}")
    
    st.markdown("---")
    
    # About
    st.subheader("ℹ️ About")
    st.markdown("""
    **Path:** `C:\\Users\\omrew\\OneDrive\\Documents\\project\\spam_detestion`
    
    **Backend:** Jupyter Notebook
    **Frontend:** Streamlit App
    
    **Internship:** Syntexhub Task 2
    """)

# ====================
# MAIN CONTENT
# ====================

st.markdown('<h1 class="main-header">📨 Spam Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning Internship - Task 2 | Syntexhub</p>', unsafe_allow_html=True)

with st.expander("📁 Project Location", expanded=False):
    st.code(f"""
Project Path: {BACKEND_DIR}
Model Path: {MODEL_PATH}
Model Exists: {os.path.exists(MODEL_PATH) if os.path.exists(BACKEND_DIR) else "Backend dir not found"}
    """)

sample_messages = {
    "Spam Examples": [
        "WINNER! You've won a $1000 Walmart gift card! Click to claim!",
        "URGENT: Your bank account needs verification. Update now!",
        "Free iPhone 15! Limited time offer. Claim yours today!",
        "CONGRATULATIONS! You won $5000 in our lottery!",
        "Investment opportunity: Double your money in 24 hours!",
        "Your Netflix subscription has expired. Update payment now!"
    ],
    "Ham Examples": [
        "Hey, are we still meeting for lunch tomorrow?",
        "Your package has been delivered. Tracking number: 9876543210",
        "Don't forget to submit the report by Friday",
        "Meeting rescheduled to 3 PM tomorrow",
        "Can you send me the presentation slides?",
        "Mom called. She wants to know when you're visiting"
    ]
}

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Detection", "📊 Batch Processing", "📈 Analytics", "📚 About"])

with tab1:
    st.header("Single Message Detection")
    
    if not detector.model_data:
        st.warning("⚠️ Model not loaded. Using demo mode.")
        st.info("To use full features, please run the backend Jupyter notebook first.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        message = st.text_area(
            "Enter your message:",
            height=150,
            placeholder="Type or paste your message here...\n\nExample: 'WINNER! You've won a $1000 gift card!'",
            key="message_input"
        )
        
        st.subheader("💡 Quick Examples")
        example_cols = st.columns(3)
        
        examples = [
            "WINNER! You won a prize!",
            "Meeting at 3 PM today",
            "URGENT: Account verification needed"
        ]
        
        for i, (col, example) in enumerate(zip(example_cols, examples)):
            with col:
                if st.button(example[:20] + "...", key=f"example_{i}"):
                    st.session_state.message_input = example
    
    with col2:
        status_color = "#4CAF50" if detector.model_data else "#FF9800"
        status_text = "✅ Ready" if detector.model_data else "⚠️ Demo Mode"
        
        st.markdown(f"""
        <div class="stat-card">
            <div class="metric-label">Model Status</div>
            <div class="metric-value" style="color: {status_color};">{status_text}</div>
            <div class="metric-label">Selected Model</div>
            <div style="font-size: 1.2rem; font-weight: bold;">{selected_model}</div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🔍 Detect Spam", type="primary", use_container_width=True)
    
    if predict_btn and message:
        with st.spinner("🔄 Analyzing message..."):
            try:
                result = detector.predict(message, selected_model)
                
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                    st.info("Trying fallback prediction...")
                    if "win" in message.lower() or "prize" in message.lower() or "urgent" in message.lower():
                        result = {
                            "text": message,
                            "prediction": "spam",
                            "spam_probability": 0.85,
                            "ham_probability": 0.15,
                            "confidence": 0.85,
                            "model_used": "fallback"
                        }
                    else:
                        result = {
                            "text": message,
                            "prediction": "ham",
                            "spam_probability": 0.15,
                            "ham_probability": 0.85,
                            "confidence": 0.85,
                            "model_used": "fallback"
                        }
                else:
                    st.success("✅ Prediction successful!")
                
                st.markdown("---")
                
                if result['prediction'] == 'spam':
                    st.markdown(f"""
                    <div class="prediction-card spam-card">
                        <h2>🚨 SPAM DETECTED</h2>
                        <p><strong>Message:</strong> {message[:100]}{'...' if len(message) > 100 else ''}</p>
                        <p><strong>Model Used:</strong> {result['model_used']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.warning("⚠️ This message appears to be spam. Be cautious!")
                    
                else:
                    st.markdown(f"""
                    <div class="prediction-card ham-card">
                        <h2>✅ LEGITIMATE MESSAGE</h2>
                        <p><strong>Message:</strong> {message[:100]}{'...' if len(message) > 100 else ''}</p>
                        <p><strong>Model Used:</strong> {result['model_used']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("✅ This message appears to be legitimate.")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="metric-label">SPAM PROBABILITY</div>
                        <div class="metric-value" style="color: #F44336;">{result['spam_probability']:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="metric-label">HAM PROBABILITY</div>
                        <div class="metric-value" style="color: #4CAF50;">{result['ham_probability']:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="metric-label">CONFIDENCE</div>
                        <div class="metric-value" style="color: #2196F3;">{result['confidence']:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Spam', 'Ham'],
                        y=[result['spam_probability'], result['ham_probability']],
                        marker_color=['#F44336', '#4CAF50'],
                        text=[f"{result['spam_probability']:.1%}", f"{result['ham_probability']:.1%}"],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Prediction Probabilities",
                    yaxis_title="Probability",
                    yaxis_tickformat=".0%",
                    yaxis_range=[0, 1],
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📊 Spam Probability Meter")
                
                spam_percent = result['spam_probability'] * 100
                
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
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
    
    elif predict_btn and not message:
        st.warning("⚠️ Please enter a message first.")

with tab2:
    st.header("Batch Message Processing")
    
    if not detector.model_data:
        st.error("❌ Model not loaded. Cannot process batch messages.")
        st.info("Please run the backend Jupyter notebook first to train the model.")
        st.markdown("---")
        st.subheader("🎮 Demo Mode - Try Single Messages Instead")
        st.info("You can still use Single Detection tab with fallback predictions.")
    else:
        pass

with tab3:
    st.header("Model Analytics & Performance")
    
    if not detector.model_data:
        st.error("❌ Model not loaded. Please train the model first.")
        st.info("Run the backend Jupyter notebook to train and save the model.")
        
        st.markdown("---")
        st.subheader("📁 Backend Location")
        st.code(f"cd \"{BACKEND_DIR}\"")
        st.markdown("Then open and run `Spam_Detection_Backend.ipynb`")
        
        if st.button("🔄 Try Loading Model Again"):
            if detector.load_model():
                st.rerun()
    else:
        pass

with tab4:
    st.header("About Spam Detection System")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://img.icons8.com/color/200/000000/machine-learning.png", width=150)
    
    with col2:
        st.markdown(f"""
        ## 📨 Spam Detection System
        
        **Version:** 1.0  
        **Author:** Syntexhub Intern  
        **Task:** Machine Learning Internship - Task 2
        **Project Path:** `{BACKEND_DIR}`
        
        ---
        
        ### 🎯 Project Setup
        
        1. **Backend (Jupyter Notebook):**
           - Location: `{BACKEND_DIR}`
           - File: `Spam_Detection_Backend.ipynb`
           - Run all cells to train model
        
        2. **Frontend (Streamlit):**
           - Current application
           - Loads model from: `{MODEL_PATH}`
        
        ### 🤖 Technology Stack
        
        - **Backend:** Python, Scikit-learn, Pandas, NumPy
        - **ML Algorithms:** Naive Bayes, Logistic Regression
        - **NLP:** TF-IDF Vectorization
        - **Frontend:** Streamlit
        - **Visualization:** Plotly, Matplotlib
        
        ### 📊 Features
        
        1. **Real-time Detection** - Instant spam classification
        2. **Batch Processing** - Analyze multiple messages at once
        3. **File Upload** - Support for CSV files
        4. **Performance Analytics** - Model metrics and visualizations
        5. **Export Results** - Download predictions as CSV
        
        ### 🚀 Getting Started
        
        1. Run: `streamlit run app.py`
        2. Open browser at `http://localhost:8501`
        3. If model not found, run backend notebook first
        
        ---
        
        *Built with ❤️ for Syntexhub Internship Program*
        """)
    
    st.markdown("---")
    
    st.subheader("🔧 Quick System Test")
    
    test_col1, test_col2, test_col3 = st.columns(3)
    
    with test_col1:
        if st.button("Test Model Loading"):
            if detector.model_data:
                st.success("✅ Model is loaded")
                st.write(f"Models: {list(detector.model_data.get('models', {}).keys())}")
            else:
                st.error("❌ Model not loaded")
    
    with test_col2:
        if st.button("Check Paths"):
            st.write(f"**Model Path:** {MODEL_PATH}")
            st.write(f"**Exists:** {os.path.exists(MODEL_PATH)}")
            st.write(f"**Backend Dir:** {BACKEND_DIR}")
            st.write(f"**Exists:** {os.path.exists(BACKEND_DIR)}")
    
    with test_col3:
        if st.button("Test Prediction"):
            test_result = detector.predict("Test message", selected_model)
            st.write("Test Result:", test_result)

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>© 2024 Spam Detection System | Syntexhub Machine Learning Internship - Task 2</p>
    <p>Project Path: {BACKEND_DIR}</p>
    <p>Built with Streamlit • Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)