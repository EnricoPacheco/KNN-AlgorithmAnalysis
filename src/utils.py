import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    
    if 'binaryClass' in df.columns:                                   #se tiver uma coluna binaryClass ela é necessariamente o target
        target_col = 'binaryClass'
    else:
        target_col = df.columns[-1]
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))  #faz cada label e transforma em numero
            
    if not pd.api.types.is_numeric_dtype(y):
        y = LabelEncoder().fit_transform(y.astype(str))
        
    X = X.fillna(X.mean(numeric_only=True))                            #preenche buracos
    X = X.replace([np.inf, -np.inf], 0)                                #tira valores infinitos
        
    scaler = MinMaxScaler()                                            #normalização
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, np.array(y)