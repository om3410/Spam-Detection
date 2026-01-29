
import pickle

class SpamDetector:
    def __init__(self, model_path='spam_model.pkl'):
        with open(model_path, 'rb') as f:
            self.model_data = pickle.load(f)
    
    def predict(self, text, model='naive_bayes'):
        text_vec = self.model_data['vectorizer'].transform([text])
        prediction = self.model_data['models'][model].predict(text_vec)[0]
        probabilities = self.model_data['models'][model].predict_proba(text_vec)[0]
        
        return {
            "prediction": "spam" if prediction == 1 else "ham",
            "spam_probability": float(probabilities[1]),
            "ham_probability": float(probabilities[0])
        }
