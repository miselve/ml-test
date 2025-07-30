
# In[1]:



# In[3]:


import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# === 1. Load Dataset ===
#dataset_name = "Subscription_Service_Churn_Dataset"
dataset_name = "WA_Fn-UseC_-Telco-Customer-Churn"


# In[4]:


#load saved folds for Cross Validation
folds = joblib.load(f"export/{dataset_name}_cv_folds.pkl")


# In[5]:


#load from disk hold out
holdout = pd.read_csv(f"export/{dataset_name}_holdout_with_labels.csv")

X_holdout = holdout.drop(columns=["Churn"])  # Replace "label" with your actual label column name
y_holdout = holdout["Churn"]


# In[8]:


#load full training from disk
train = pd.read_csv(f"export/{dataset_name}_train_with_labels.csv")

X_train_full = train.drop(columns=["Churn"])   # assuming "Churn" is the label column
y_train_full = train["Churn"]


# In[9]:


#time
from datetime import datetime

now = datetime.now()
# Get current date and time (formatted)
date_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

print("Date and Time:", date_time)

# Define folder path
folder_path = f"export/{date_time}"

# Create the folder
os.makedirs(folder_path, exist_ok=True)

print("Folder created:", folder_path)


# In[11]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# === 2. Define models as a list of tuples ===
models = [
    ("RandomForest", RandomForestClassifier(random_state=42)),
    ("DecisionTree", DecisionTreeClassifier(random_state=42)),
    ("LogisticRegression", LogisticRegression(random_state=42, max_iter=1000)),
    ("SVM", SVC(probability=True, random_state=42)),
    ("NaiveBayes", GaussianNB()),
    ("MLP", MLPClassifier(random_state=42, max_iter=300)),
    ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)),
    ("LightGBM", LGBMClassifier(random_state=42))
]
#loop to run for each model


# In[12]:


results_cross = []
results_full = []


# In[13]:


#Cross-Validation


# In[ ]:


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

import psutil

for model_name, model_cv in models:

    import time
    from codecarbon import OfflineEmissionsTracker
    
    #from codecarbon.core.schemas import GeoLocation
    tracker = OfflineEmissionsTracker(
        country_iso_code="GRC",  # Greece
        output_dir=folder_path,
        output_file=f"{dataset_name}_{model_name}_cross_emissions.csv",
        save_to_file=True
    )
    
    tracker.start()
    
    start_time_cross = time.perf_counter()

    # CPU and RAM snapshot
    cpu_before = psutil.cpu_percent()
    ram_before = psutil.virtual_memory().used
    
    cv_results = cross_validate(
        model_cv, X_train_full, y_train_full,
        cv=folds,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        return_train_score=False
    )
    
    # === Stop tracking and get emissions ===
    #emissions = tracker.stop()
    end_time_cross = time.perf_counter()
    
    training_time_cross_validation = end_time_cross - start_time_cross
    
    print("=== Cross-Validation Performance ===")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        print(f"Mean {metric}: {cv_results[f'test_{metric}'].mean():.4f}")
    
    # Print results
    #print(f"Total emissions during 10-fold CV: {emissions * 1000:.3f} g CO₂")

        # CPU and RAM snapshot after
    cpu_after = psutil.cpu_percent()
    ram_after = psutil.virtual_memory().used

    
    tracker.stop()
    import pandas as pd
    log_file = f"{folder_path}/{dataset_name}_{model_name}_cross_emissions.csv"
    #log_file = f"export/{dataset_name}_{model_name}_cross_emissions.csv"
    df = pd.read_csv(log_file)
    emissions_cross = df.iloc[-1]["emissions"]
    print(f"Total emissions during 10-fold CV: {emissions_cross * 1000:.3f} g CO₂")
    
    print(f"Total training time 10-fold CV: {training_time_cross_validation:.3f} sec")

    results_cross.append({
    "Model": model_name,
    "Mean Accuracy": cv_results['test_accuracy'].mean(),
    "Mean Precision": cv_results['test_precision'].mean(),
    "Mean Recall": cv_results['test_recall'].mean(),
    "Mean F1-score": cv_results['test_f1'].mean(),
    "Mean AUC (CV)": cv_results['test_roc_auc'].mean(),
    "Total Training Time CV (s)": training_time_cross_validation,
    "Total Emissions CV (kgCO₂)": emissions_cross,
    "RAM Usage During Training (GB)": ram_after - ram_before
    })

    


# In[ ]:


# === 4. Display Results ===
results_df_cross = pd.DataFrame(results_cross)


# In[ ]:


results_df_cross.to_csv(folder_path+"/"+"Results_10-fold_cross_"+dataset_name+"_"+".csv", index=False)


# In[25]:


for model_name, model_full in models:
    
        # === 3. Retrain final model on full training data ===
    tracker_full_training = OfflineEmissionsTracker(
        country_iso_code="GRC",  # Greece
        output_dir=folder_path,
        output_file=f"{dataset_name}_{model_name}_full_training_emissions.csv",
        save_to_file=True
    )
    
    tracker_full_training.start()
    
    start_time = time.perf_counter()

     # CPU and RAM snapshot
    cpu_before_full = psutil.cpu_percent()
    ram_before_full = psutil.virtual_memory().used
    
    
    model_full.fit(X_train_full, y_train_full)
    
    end_time = time.perf_counter()

         # CPU and RAM snapshot after
    cpu_after_full = psutil.cpu_percent()
    ram_after_full = psutil.virtual_memory().used

    
    emissions_full = tracker_full_training.stop()
    
    training_time_full = end_time - start_time
    
    print(f"Training time full: {training_time_full:.3f} sec")
    print(f"CO₂ emitted training full: {emissions_full * 1000:.3f} g")

    
        # === 4. Evaluate on holdout set ===
    tracker_holdout = OfflineEmissionsTracker(
        country_iso_code="GRC",  # Greece
        output_dir=folder_path,
        output_file=f"{dataset_name}_{model_name}_hold_out_emissions.csv",
        save_to_file=True
    )
    tracker_holdout.start()
    
    start_time = time.perf_counter()

     # CPU and RAM snapshot
    cpu_before_hold_out = psutil.cpu_percent()
    ram_before_hold_out = psutil.virtual_memory().used
    
    y_pred = model_full.predict(X_holdout)
    y_proba = model_full.predict_proba(X_holdout)[:, 1]

        # CPU and RAM snapshot after
    cpu_after_holdout = psutil.cpu_percent()
    ram_after_holdout = psutil.virtual_memory().used
    
    end_time = time.perf_counter()

    
    tracker_holdout.stop()
    
    # Read emissions from file
    import pandas as pd
    log_file = f"{folder_path}/{dataset_name}_{model_name}_hold_out_emissions.csv"
    df = pd.read_csv(log_file)
    emissions_holdout = df.iloc[-1]["emissions"]
    
    #print(f"CO₂ emitted prediction: {emissions * 1000:.3f} g")
    
    prediction_time_hold_out = end_time - start_time
    
    n_instances = len(X_holdout)
    
    mean_prediction_time_hold_out = prediction_time_hold_out/n_instances
    
    print(f"Prediction time: {prediction_time_hold_out:.3f} sec")
    print(f"CO₂ emitted prediction: {emissions_holdout * 1000:.3f} g")
    
    print("\n=== Final Evaluation on Holdout Set ===")
    print(classification_report(y_holdout, y_pred))
    print("AUC on Holdout:", roc_auc_score(y_holdout, y_proba))
    
    from sklearn.metrics import precision_recall_fscore_support
    # Precision, Recall, F1
    precision_hold_out, recall_hold_out, f1_hold_out, _ = precision_recall_fscore_support(y_holdout, y_pred, average='binary')
    print(f"Precision: {precision_hold_out:.2f}, Recall: {recall_hold_out:.2f}, F1-score: {f1_hold_out:.2f}")
    
    from sklearn.metrics import accuracy_score
    
    accuracy_hold_out = accuracy_score(y_holdout, y_pred)
    print(f"Accuracy: {accuracy_hold_out:.2f}")

        # === 6. Export Model and Features ===
    #os.makedirs("export", exist_ok=True)
    joblib.dump(model_full, folder_path+"/"+model_name+"_full.pkl")
    
    model_size_full = os.path.getsize(folder_path+"/"+model_name+"_full.pkl") / 1024**2  # in MB
    
    print("Model size: ", model_size_full)
    
    
    results_full.append({
            "Model": model_name,
            "Accuracy hold out":accuracy_hold_out,
            "AUC": roc_auc_score(y_holdout, y_pred),
            "Precision":precision_hold_out,
            "Recall":recall_hold_out,
            "F-measure":f1_hold_out,
            "Training Time (s)": training_time_full,
            "Prediction Time (s)": prediction_time_hold_out,
            "Mean Prediction Time (s)": mean_prediction_time_hold_out,
            "Model Size (MB)": model_size_full,
            "Emissions Prediction hold out ": emissions_holdout,
            "Emissions Training": emissions_full,
            "RAM Usage During Training (GB)": ram_after_full - ram_before_full,
            "RAM Usage During Prediction (GB)": ram_after_holdout - ram_before_holdout
        })
    
    
        


# In[26]:


# === 4. Display Results ===
results_df_full = pd.DataFrame(results_full)
print(results_df_full.to_string(index=False))

results_df_full.to_csv(folder_path+"/"+"Results_full_training_"+dataset_name+"_"+".csv", index=False)


# In[ ]:




