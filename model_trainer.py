import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

class HealthModelTrainer:
    def __init__(self, test_size=0.2, max_features=5000, random_state=42):
        self.test_size = test_size
        self.max_features = max_features
        self.random_state = random_state
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.9,  # Ignore terms that appear in more than 90% of documents
            sublinear_tf=True  # Use sublinear TF scaling
        )
        self.classifier = LogisticRegression(
            random_state=random_state, 
            max_iter=1000,
            C=1.0,  # Regularization strength
            class_weight='balanced'  # Handle class imbalance
        )
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        
        # Model state
        self.is_trained = False
        self.feature_names = None
        self.classes = None
        self.training_metrics = {}
        
    def preprocess_text(self, text):
        """Preprocess text for training and prediction"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        stop_words = set(stopwords.words('english'))
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)
    
    def train(self, data):
        """Train the model on processed data"""
        try:
            # Preprocess questions
            preprocessed_questions = [self.preprocess_text(q) for q in data['questions']]
            
            # Encode responses
            encoded_responses = self.label_encoder.fit_transform(data['responses'])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                preprocessed_questions, 
                encoded_responses,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=encoded_responses
            )
            
            # Vectorize text
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)
            
            # Train classifier
            self.classifier.fit(X_train_tfidf, y_train)
            
            # Make predictions
            y_pred = self.classifier.predict(X_test_tfidf)
            y_pred_proba = self.classifier.predict_proba(X_test_tfidf)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate additional metrics
            from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            # Store training metrics
            self.training_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'num_features': X_train_tfidf.shape[1],
                'num_classes': len(self.label_encoder.classes_),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'feature_names': self.vectorizer.get_feature_names_out()[:50].tolist(),  # Store top 50 features
                'class_names': self.label_encoder.classes_.tolist()
            }
            
            # Store model state
            self.is_trained = True
            self.feature_names = self.vectorizer.get_feature_names_out()
            self.classes = self.label_encoder.classes_
            
            return self.training_metrics
            
        except Exception as e:
            raise Exception(f"Training failed: {str(e)}")
    
    def predict(self, text):
        """Predict response for given text"""
        if not self.is_trained:
            raise Exception("Model not trained yet")
        
        # Preprocess text
        preprocessed = self.preprocess_text(text)
        
        # Vectorize
        text_tfidf = self.vectorizer.transform([preprocessed])
        
        # Predict
        prediction = self.classifier.predict(text_tfidf)[0]
        probabilities = self.classifier.predict_proba(text_tfidf)[0]
        
        # Get response and confidence
        response = self.label_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        return response, confidence
    
    def get_similar_responses(self, text, top_k=3):
        """Get top-k similar responses with confidence scores"""
        if not self.is_trained:
            raise Exception("Model not trained yet")
        
        # Preprocess text
        preprocessed = self.preprocess_text(text)
        
        # Vectorize
        text_tfidf = self.vectorizer.transform([preprocessed])
        
        # Get probabilities for all classes
        probabilities = self.classifier.predict_proba(text_tfidf)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            response = self.label_encoder.inverse_transform([idx])[0]
            confidence = probabilities[idx]
            results.append((response, confidence))
        
        return results
    
    def save_model(self, filepath):
        """Save trained model to file"""
        if not self.is_trained:
            raise Exception("Cannot save untrained model")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'lemmatizer': self.lemmatizer,
            'training_metrics': self.training_metrics,
            'classes': self.classes,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance
        instance = cls()
        
        # Restore model state
        instance.vectorizer = model_data['vectorizer']
        instance.classifier = model_data['classifier']
        instance.label_encoder = model_data['label_encoder']
        instance.lemmatizer = model_data['lemmatizer']
        instance.training_metrics = model_data['training_metrics']
        instance.classes = model_data['classes']
        instance.feature_names = model_data['feature_names']
        instance.is_trained = model_data['is_trained']
        
        return instance
    
    def get_feature_importance(self, top_n=10):
        """Get most important features for each class"""
        if not self.is_trained:
            raise Exception("Model not trained yet")
        
        feature_importance = {}
        
        for i, class_name in enumerate(self.classes):
            # Get feature weights for this class
            weights = self.classifier.coef_[i]
            
            # Get top positive and negative features
            top_positive_idx = np.argsort(weights)[-top_n:][::-1]
            top_negative_idx = np.argsort(weights)[:top_n]
            
            feature_importance[class_name] = {
                'positive': [(self.feature_names[idx], weights[idx]) for idx in top_positive_idx],
                'negative': [(self.feature_names[idx], weights[idx]) for idx in top_negative_idx]
            }
        
        return feature_importance
