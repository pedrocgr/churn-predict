import os
import glob
import pandas as pd
import numpy as np
import torch
import optuna
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from models.mitra import MitraClassifier

# Setup
RANDOM_STATE = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_TRIALS = 25  # Número de tentativas do Optuna

print(f"⚡ Rodando MITRA (Transformer) no dispositivo: {DEVICE}")

# Loading
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    csv_path = 'customer_churn_telecom_services.csv'
else:
    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    csv_path = csv_files[0] if csv_files else 'customer_churn_telecom_services.csv'

print(f"Dataset: {csv_path}")
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print("Erro: CSV não encontrado.")
    exit()

# Cleaning & Preprocessing 
target_col = "Churn"
def to_binary(series):
    if series.dtype == 'O':
        return series.str.lower().map({'yes':1,'sim':1,'true':1,'no':0,'nao':0}).fillna(series)
    return series

df[target_col] = to_binary(df[target_col])
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

categorical_cols = [c for c in X.columns if X[c].dtype == 'O']
numeric_cols = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

print("Processando dados...")
X_processed = preprocessor.fit_transform(X)

# Data Splitting
# A: Separar TESTE (25%) - Intocável
X_temp, X_test, y_temp, y_test = train_test_split(
    X_processed, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# B: Separar VALIDAÇÃO (25% do total)
# 33% do restante equivale a 25% do total
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.3333, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"Treino: {X_train.shape} | Validação: {X_val.shape} | Teste: {X_test.shape}")

# SMOTE (Augmentation) 
print("Aplicando SMOTE no Treino...")
try:
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
except Exception as e:
    print(f"Erro no SMOTE: {e}. Usando dados originais.")
    X_train_bal, y_train_bal = X_train, y_train

# Otimização Optuna 

def objective(trial):
    # Hiperparâmetros do Transformer
    d_model = trial.suggest_categorical('d_model', [32, 64, 128])
    nhead = trial.suggest_categorical('nhead', [2, 4])
    
    # Regra do PyTorch: d_model deve ser divisível por nhead
    if d_model % nhead != 0:
        raise optuna.TrialPruned()

    num_layers = trial.suggest_int('num_layers', 1, 3)
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    
    model = MitraClassifier(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        learning_rate=lr,
        batch_size=batch_size,
        max_epochs=15, # Teste rápido
        device=DEVICE
    )
    
    model.fit(X_train_bal, y_train_bal)
    
    # Avaliação na Validação
    probs_val = model.predict_proba(X_val)[:, 1]
    
    try:
        return roc_auc_score(y_val, probs_val)
    except:
        return 0.5

print("\nOptuna: Buscando melhores parâmetros...")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS)

print(f"\nMelhores Parâmetros: {study.best_params}")

# Treinamento Final com os Melhores Parâmetros 
print("\nTreinando MITRA Final...")
bp = study.best_params

final_model = MitraClassifier(
    d_model=bp['d_model'],
    nhead=bp['nhead'],
    num_layers=bp['num_layers'],
    dim_feedforward=bp['dim_feedforward'],
    dropout=bp['dropout'],
    learning_rate=bp['lr'],
    batch_size=bp['batch_size'],
    max_epochs=60, # Treino longo para convergência
    device=DEVICE
)

final_model.fit(X_train_bal, y_train_bal)

print("\nRESULTADOS FINAIS - MITRA (AMAZON)")
probs_test = final_model.predict_proba(X_test)[:, 1]
preds_test = (probs_test >= 0.5).astype(int)

print(classification_report(y_test, preds_test))
print(f"AUROC Final: {roc_auc_score(y_test, probs_test):.4f}")