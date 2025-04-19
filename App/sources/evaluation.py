from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import time
import pandas as pd

def evaluate_models(test_df, rf_detector, bert_trainer, bert_tokenizer):
    results = {}
    test_texts = test_df['value'].tolist()
    true_labels = test_df['flag'].astype(int).tolist()
    
    # evaluate random forest model
    start_time = time.time()
    rf_predictions = []
    rf_probabilities = []
    
    for text in test_texts:
        prediction = rf_detector.predict(text)
        rf_predictions.append(1 if prediction["has_pii"] else 0)
        rf_probabilities.append(prediction["confidence"])
    
    rf_time = time.time() - start_time
    
    # evaluate bert model
    start_time = time.time()
    
    # prepare bert input
    import torch
    device = torch.device("cpu")  # force using CPU
    
    bert_inputs = bert_tokenizer(test_texts, padding=True, truncation=True, 
                                 max_length=128, return_tensors="pt")
    
    # move input to CPU
    for key in bert_inputs:
        bert_inputs[key] = bert_inputs[key].to(device)
    
    # ensure model on CPU
    bert_trainer.model = bert_trainer.model.to(device)
    
    # get bert prediction
    with torch.no_grad():
        bert_outputs = bert_trainer.model(**bert_inputs)
        bert_probabilities = torch.nn.functional.softmax(bert_outputs.logits, dim=-1)[:, 1].cpu().numpy()
    
    bert_predictions = [1 if prob > 0.5 else 0 for prob in bert_probabilities]
    bert_time = time.time() - start_time
    
    # calculate metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'inference_time']
    
    results['random_forest'] = {
        'accuracy': accuracy_score(true_labels, rf_predictions),
        'precision': precision_score(true_labels, rf_predictions),
        'recall': recall_score(true_labels, rf_predictions),
        'f1': f1_score(true_labels, rf_predictions),
        'auc': roc_auc_score(true_labels, rf_probabilities),
        'inference_time': rf_time
    }
    
    results['bert'] = {
        'accuracy': accuracy_score(true_labels, bert_predictions),
        'precision': precision_score(true_labels, bert_predictions),
        'recall': recall_score(true_labels, bert_predictions),
        'f1': f1_score(true_labels, bert_predictions),
        'auc': roc_auc_score(true_labels, bert_probabilities),
        'inference_time': bert_time
    }
    
    return results

def analyze_case_differences(test_df, rf_detector, bert_trainer, bert_tokenizer):
    test_texts = test_df['value'].tolist()
    true_labels = test_df['flag'].astype(int).tolist()
    
    # get rf and bert prediction
    rf_results = []
    bert_results = []
    
    for text in test_texts:
        # rf prediction
        rf_pred = rf_detector.predict(text)
        rf_results.append({
            'prediction': 1 if rf_pred["has_pii"] else 0,
            'confidence': rf_pred["confidence"],
            'details': rf_pred.get("details", {})
        })
        
    # bert prediction
    import torch
    device = torch.device("cpu")  # force using CPU
    bert_inputs = bert_tokenizer(test_texts, padding=True, truncation=True, 
                               max_length=128, return_tensors="pt")
    for key in bert_inputs:
        bert_inputs[key] = bert_inputs[key].to(device)
    
    bert_trainer.model = bert_trainer.model.to(device)


    with torch.no_grad():
        bert_outputs = bert_trainer.model(**bert_inputs)
        bert_probs = torch.nn.functional.softmax(bert_outputs.logits, dim=-1)[:, 1].cpu().numpy()
    
    for i, prob in enumerate(bert_probs):
        bert_results.append({
            'prediction': 1 if prob > 0.5 else 0,
            'confidence': float(prob)
        })
    
    # find disagreements between rf and bert
    disagreements = []
    for i, (text, true_label, rf_result, bert_result) in enumerate(zip(test_texts, true_labels, rf_results, bert_results)):
        if rf_result['prediction'] != bert_result['prediction']:
            disagreements.append({
                'index': i,
                'text': text,
                'true_label': true_label,
                'rf_prediction': rf_result['prediction'],
                'rf_confidence': rf_result['confidence'],
                'rf_details': rf_result.get('details', {}),
                'bert_prediction': bert_result['prediction'],
                'bert_confidence': bert_result['confidence'],
                'correct_model': 'Random Forest' if rf_result['prediction'] == true_label else 
                               ('BERT' if bert_result['prediction'] == true_label else 'Neither')
            })
    
    return pd.DataFrame(disagreements)




