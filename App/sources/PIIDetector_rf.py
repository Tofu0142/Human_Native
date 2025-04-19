import spacy
import re
from typing import Dict, List, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from .data_class import Data
from ..data.data_generation import generate_fake_dataset
import pandas as pd

class PIIDetector:
    def __init__(self):
        # load spacy model
        self.nlp = spacy.load("en_core_web_lg")
        
        # initialize classifier
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=5000)
        
        # compile regex patterns
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b'),
            'address': re.compile(r'\b\d+\s+[A-Za-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Court|Ct|Lane|Ln|Way)\b', re.IGNORECASE)
        }
        
        # store verification status in memory (not modify data model)
        self.verification_status = {}  # {data_id: status}
        
    def rule_based_detection(self, text: str) -> Dict[str, List[str]]:
        """use rules and NER to detect PII"""
        results = {}
        
        # regex detection
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                results[pii_type] = matches
        
        # NER detection
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
                pii_type = ent.label_.lower()
                if pii_type not in results:
                    results[pii_type] = []
                results[pii_type].append(ent.text)
                
        return results
    
    def train(self, training_data):
        """train classifier"""
        # check data type and process accordingly
        if isinstance(training_data, pd.DataFrame):
            # process DataFrame
            print(f"Training with DataFrame containing {len(training_data)} rows")
            print(f"DataFrame columns: {training_data.columns.tolist()}")
            print(f"First row sample: {training_data.iloc[0].to_dict()}")
            
            # ensure DataFrame contains necessary columns
            if 'value' in training_data.columns and 'flag' in training_data.columns:
                texts = training_data['value'].tolist()
                labels = [1 if flag else 0 for flag in training_data['flag'].tolist()]
                print(f"Extracted {len(texts)} texts and {len(labels)} labels")
            else:
                raise ValueError(f"DataFrame must contain 'value' and 'flag' columns. Found: {training_data.columns.tolist()}")
        elif len(training_data) > 0:
            if hasattr(training_data[0], 'value'):  # if Data object
                texts = [item.value for item in training_data]
                labels = [1 if item.flag else 0 for item in training_data]
            elif isinstance(training_data[0], dict):  # if dictionary
                texts = [item['value'] for item in training_data]
                labels = [1 if item['flag'] else 0 for item in training_data]
            elif isinstance(training_data[0], str):  # if string
                texts = training_data
                labels = [0] * len(texts)  # default all labels are 0
                print("Warning: No labels provided, assuming all negative")
            else:
                raise TypeError(f"Unsupported training data format: {type(training_data[0])}")
        else:
            texts = []
            labels = []
            print("Warning: Empty training data")
        
        # convert texts to feature vectors
        print(f"Converting {len(texts)} texts to feature vectors")
        X = self.vectorizer.fit_transform(texts)
        
        # split data into training and testing
        if len(texts) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
            print(f"Split data into {X_train.shape[0]} training and {X_test.shape[0]} testing samples")
            
            # train classifier
            self.classifier.fit(X_train, y_train)
            print("Model training completed")
            
            # evaluate model
            y_pred = self.classifier.predict(X_test)
            print("\nModel Evaluation:")
            print(classification_report(y_test, y_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
        else:
            print("Not enough data to train the model")
        
    def predict(self, text: str) -> Dict[str, Any]:
        """predict if text contains PII"""
        # rule detection
        rule_results = self.rule_based_detection(text)
        
        # ml prediction
        X = self.vectorizer.transform([text])
        ml_prediction = self.classifier.predict_proba(X)[0][1]  # get positive class probability
        
        # combine results
        has_pii = len(rule_results) > 0 or ml_prediction > 0.5
        confidence = ml_prediction
        if len(rule_results) > 0:
            confidence = max(confidence, 0.8)  # if rule detection found PII, increase confidence
            
        return {
            "has_pii": has_pii,
            "confidence": confidence,
            "pii_types": list(rule_results.keys()),
            "details": rule_results
        }
    
    def process_data_item(self, data_item: Data) -> Data:
        """process data item and update flag"""
        result = self.predict(data_item.value)
        # only update flag field, not add new fields
        data_item.flag = result["has_pii"]
        
        # store extra info in memory (not modify data model)
        self.verification_status[data_item.id] = {
            "confidence": result["confidence"],
            "pii_types": result["pii_types"],
            "details": result["details"]
        }
        
        return data_item


