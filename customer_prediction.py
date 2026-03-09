"""
Customer Cancellation Prediction Using Machine Learning
====================================================
This project predicts which customers will STOP using the company services.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. GENERATE COMPANY DATA
# ============================================

def generate_company_data(n_customers=1000):
    """Generate customer data for ONE company"""
    np.random.seed(42)
    
    data = {
        'customer_id': range(1, n_customers + 1),
        'age': np.random.randint(18, 80, n_customers),
        'months_with_company': np.random.randint(1, 73, n_customers),
        'monthly_bill': np.random.randint(20, 150, n_customers),
        'total_paid': np.random.randint(20, 12000, n_customers),
        'support_calls': np.random.randint(0, 7, n_customers),
    }
    
    # Contract: 0=Monthly, 1=1-Year, 2=2-Year
    data['contract'] = np.random.choice([0, 1, 2], n_customers, p=[0.55, 0.25, 0.20])
    
    # Internet: 0=None, 1=DSL, 2=Fiber
    data['internet'] = np.random.choice([0, 1, 2], n_customers, p=[0.22, 0.34, 0.44])
    
    # Extra services
    data['has_security'] = np.random.choice([0, 1], n_customers, p=[0.65, 0.35])
    data['has_support_plan'] = np.random.choice([0, 1], n_customers, p=[0.72, 0.28])
    
    # Calculate cancellation probability based on risk factors
    cancel_prob = np.zeros(n_customers)
    cancel_prob += (data['contract'] == 0) * 0.25      # Monthly = high risk
    cancel_prob += (data['months_with_company'] < 12) * 0.20  # New customers
    cancel_prob += (data['support_calls'] > 3) * 0.15  # Many complaints
    cancel_prob += (data['has_security'] == 0) * 0.08
    cancel_prob += (data['has_support_plan'] == 0) * 0.08
    cancel_prob += (data['internet'] == 2) * 0.05
    
    # Add randomness
    cancel_prob = cancel_prob / cancel_prob.max() + np.random.uniform(-0.1, 0.1, n_customers)
    cancel_prob = np.clip(cancel_prob, 0, 1)
    
    # Determine if customer cancelled (stopped using)
    data['cancelled'] = (np.random.random(n_customers) < cancel_prob).astype(int)
    
    return pd.DataFrame(data)

# ============================================
# 2. DISPLAY COMPANY DATA SIMPLY
# ============================================

def display_company_info(df):
    """Show company data in simple terms"""
    print("\n" + "="*50)
    print("COMPANY CUSTOMER DATA")
    print("="*50)
    
    total = len(df)
    cancelled = df['cancelled'].sum()
    active = total - cancelled
    
    print("\nTotal Customers:", total)
    print("Customers who STOPPED using us:", cancelled, "(" + str(round(cancelled/total*100, 1)) + "%)")
    print("Customers who STAYED with us:", active, "(" + str(round(active/total*100, 1)) + "%)")
    
    print("\n--- Customer Profile ---")
    print("Average Age:", round(df['age'].mean(), 0), "years")
    print("Average Time with Us:", round(df['months_with_company'].mean(), 0), "months")
    print("Average Monthly Bill: $" + str(round(df['monthly_bill'].mean(), 0)))
    print("Average Support Calls:", round(df['support_calls'].mean(), 1))
    
    print("\n--- Cancellation Rate by Contract ---")
    contract_names = {0: "Monthly", 1: "1-Year", 2: "2-Year"}
    for ctype in [0, 1, 2]:
        subset = df[df['contract'] == ctype]
        rate = subset['cancelled'].mean() * 100
        print("  " + contract_names[ctype] + ": " + str(round(rate, 1)) + "% cancelled")
    
    print("\n--- Cancellation Rate by Internet ---")
    internet_names = {0: "No Internet", 1: "DSL", 2: "Fiber"}
    for itype in [0, 1, 2]:
        subset = df[df['internet'] == itype]
        rate = subset['cancelled'].mean() * 100
        print("  " + internet_names[itype] + ": " + str(round(rate, 1)) + "% cancelled")

# ============================================
# 3. TRAIN MODELS
# ============================================

def train_models(X_train, X_test, y_train, y_test):
    """Train prediction models"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=50),
        'Support Vector Machine': SVC(kernel='rbf', probability=True)
    }
    
    results = {}
    
    print("\n" + "="*50)
    print("MODEL RESULTS")
    print("="*50)
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {'model': model, 'accuracy': acc, 'predictions': y_pred}
        
        print("\n" + name + ":")
        print("  Correct Predictions: " + str(round(acc*100, 1)) + "%")
        print("  Precision: " + str(round(prec*100, 1)) + "%")
        print("  Recall: " + str(round(rec*100, 1)) + "%")
        print("  F1 Score: " + str(round(f1*100, 1)) + "%")
    
    return results

# ============================================
# 4. SHOW BEST PREDICTOR
# ============================================

def show_best_predictor(results, y_test):
    """Show which model worked best"""
    print("\n" + "="*50)
    print("BEST PREDICTION MODEL")
    print("="*50)
    
    best_name = max(results, key=lambda x: results[x]['accuracy'])
    best = results[best_name]
    
    print("\nModel: " + best_name)
    print("Accuracy: " + str(round(best['accuracy']*100, 1)) + "%")
    
    cm = confusion_matrix(y_test, best['predictions'])
    print("\nPrediction Results:")
    print("  Correctly predicted STAY: " + str(cm[0,0]))
    print("  Wrongly predicted STOP: " + str(cm[0,1]))
    print("  Missed (actually STOPPED): " + str(cm[1,0]))
    print("  Correctly predicted STOP: " + str(cm[1,1]))

# ============================================
# 5. MAIN PROGRAM
# ============================================

if __name__ == "__main__":
    print("="*50)
    print("CUSTOMER CANCELLATION PREDICTION")
    print("="*50)
    
    print("\n[1] Loading company data...")
    df = generate_company_data(1000)
    print("    Loaded 1000 customer records")
    
    print("\n[2] Analyzing customer data...")
    display_company_info(df)
    
    print("\n[3] Setting up prediction...")
    features = ['age', 'months_with_company', 'monthly_bill', 'total_paid', 
                'support_calls', 'contract', 'internet', 'has_security', 'has_support_plan']
    X = df[features]
    y = df['cancelled']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("    Training: " + str(len(X_train)) + " customers")
    print("    Testing: " + str(len(X_test)) + " customers")
    
    print("\n[4] Training prediction models...")
    results = train_models(X_train, X_test, y_train, y_test)
    
    print("\n[5] Finding best model...")
    show_best_predictor(results, y_test)
    
    print("\n" + "="*50)
    print("KEY FINDINGS")
    print("="*50)
    print("\nCustomers more likely to STOP using our service:")
    print("  - Monthly contracts (no commitment)")
    print("  - New customers (less than 1 year)")
    print("  - Many support calls/complaints")
    print("  - No security or support plan")
    
    print("\nCustomers less likely to STOP:")
    print("  - Long-term contracts (1-2 years)")
    print("  - Long-time customers")
    print("  - Have extra services (security, support)")
    
    print("\n" + "="*50)
    print("DONE!")
    print("="*50)

