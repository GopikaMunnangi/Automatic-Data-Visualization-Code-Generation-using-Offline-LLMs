import os, time, gc, re, io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI  # Use 'pip install openai'
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. CONFIGURATION
# Connect to LM Studio Local Server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Model names must match exactly what you see in the LM Studio "Loaded Models" list
MAIN_MODEL = "qwen2.5-coder-3b-instruct"
REPAIR_MODEL = "deepseek-coder-1.3b-instruct"
    
def run_local_agent(prompt, model_name):
    """Sends a prompt to a specific model loaded in LM Studio."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: Could not reach LM Studio server. {e}"

# 2. INTELLIGENCE SUITE
def run_ml_suite(df):
    """Performs automated anomaly detection and clustering."""
    # Convert dates automatically
    for col in df.columns:
        if df[col].dtype == 'object':
            try: df[col] = pd.to_datetime(df[col], errors='ignore')
            except: pass
            
    num_df = df.select_dtypes(include=[np.number]).fillna(0)
    if num_df.empty:
        return {"cols": list(df.columns), "anomalies": 0, "pca_var": "0%"}
    
    # ML Logic
    scaled = StandardScaler().fit_transform(num_df)
    svm = OneClassSVM(nu=0.05).fit(scaled)
    anomalies = (svm.predict(scaled) == -1).sum()
    
    pca = PCA(n_components=min(2, num_df.shape[1])).fit(scaled)
    var = np.sum(pca.explained_variance_ratio_)
    
    kmeans = KMeans(n_clusters=min(3, len(df)), n_init=10).fit(scaled)
    df['cluster_label'] = kmeans.labels_
    
    return {"cols": list(df.columns), "anomalies": anomalies, "pca_var": f"{var:.2%}"}

# 3. REPAIR PIPELINE
def execute_and_repair(code, data, attempt=1):
    """Executes code and uses DeepSeek to fix any errors."""
    # Strip markdown and redundant read_csv calls
    lines = [l.lstrip() for l in code.split('\n') if 'read_csv' not in l and '```' not in l]
    clean_code = "\n".join(lines)
    
    try:
        # Clear previous plots to save RAM
        plt.clf(); plt.close('all')
        
        # Define the environment for execution
        exec_scope = {'df': data, 'plt': plt, 'sns': sns, 'pd': pd, 'np': np}
        exec(clean_code, exec_scope)
        plt.show() # Opens plot in a local window
        
        return clean_code, "SUCCESS"
    
    except Exception as e:
        if attempt > 2: 
            return clean_code, f"FAILED: {e}"
        
        print(f"Attempt {attempt} failed. Sending to {REPAIR_MODEL} for fix...")
        
        repair_prompt = (
            f"Fix this Python error: {e}. "
            f"Context: Dataframe 'df' has columns: {list(data.columns)}. "
            f"Return ONLY corrected code.\nCode:\n{clean_code}\nCorrected Code:"
        )
        # Use DeepSeek specifically for debugging
        repaired_code = run_local_agent(repair_prompt, REPAIR_MODEL)
        return execute_and_repair(repaired_code, data, attempt + 1)

# 4. MAIN OFFLINE LOOP
if __name__ == "__main__":
    file_path = input("Enter full path to your CSV file: ")
    
    if os.path.exists(file_path):
        current_df = pd.read_csv(file_path)
        print(f"Loaded {len(current_df)} rows. Analyzing...")
        
        # Get automated insights
        meta = run_ml_suite(current_df)
        print(f"Dataset Info: {meta['anomalies']} anomalies found. PCA Variance: {meta['pca_var']}")
        
        user_req = input("What would you like to visualize/calculate? ")
        
        # 1. Primary Generation via Qwen
        p = f"Columns: {list(current_df.columns)}. Task: {user_req}. Use 'df'. Return ONLY code in ```python blocks."
        ai_resp = run_local_agent(p, MAIN_MODEL)
        
        # 2. Extract and Run/Repair
        match = re.search(r"```python\n(.*?)```", ai_resp, re.DOTALL)
        if match:
            final_code, status = execute_and_repair(match.group(1).strip(), current_df)
            print(f"Status: {status}")
            print(f"Executed Code:\n{final_code}")
        else:
            print("AI failed to provide a valid code block.")
        
        # 3. Final Memory Cleanup
        del current_df
        gc.collect()
        print("Session memory cleared.")
        
    else:
        print("File not found. Please check the path.")