import os
import glob
import pandas as pd
import numpy as np
import torch
import optuna
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE # Data Augmentation/Balancing
from models.kan import KANClassifier

# Setup
RANDOM_STATE = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_TRIALS = 20  

print(f"Rodando no dispositivo: {DEVICE}")

# Loading
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    csv_path = 'customer_churn_telecom_services.csv' 
else:
    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    csv_path = csv_files[0] if csv_files else 'customer_churn_telecom_services.csv'

print(f"Carregando dataset: {csv_path}")
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print("Erro: Arquivo CSV não encontrado! Verifique se ele está na pasta 'data' ou na raiz.")
    exit()

# Cleaning & Preprocessing
target_col = "Churn"

# Convert Target to 0/1
def to_binary(series):
    if series.dtype == 'O':
        return series.str.lower().map({'yes':1,'sim':1,'true':1,'no':0,'nao':0,'false':0}).fillna(series)
    return series
df[target_col] = to_binary(df[target_col])

# Cleaning TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)

# Separate X & y
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# Identify columns
categorical_cols = [c for c in X.columns if X[c].dtype == 'O']
numeric_cols = [c for c in X.columns if c not in categorical_cols]

# Preprocessing (OneHot + Standardizer)
# Ajuste do preprocessor para lidar com desconhecidos
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# Apply preprocessing to EVERYTHING before the split (to simplify the Optuna loop)
print("Processando dados...")
X_processed = preprocessor.fit_transform(X)

# Divisão dos Dados 
# Split 1: Separar Teste (25%) do resto
X_temp, X_test, y_temp, y_test = train_test_split(
    X_processed, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# Split 2: Separar Validação (25% do total original --> 33% do que sobrou)
# Treino ficará com 50% do total original
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.3333, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"Dimensões: Treino={X_train.shape}, Validação={X_val.shape}, Teste={X_test.shape}")

# Data Augmentation (SMOTE) no conjunto de Treino
print("Aplicando Data Augmentation (SMOTE) no conjunto de Treino...")
try:
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"-> Antes: {np.bincount(y_train)} | Depois: {np.bincount(y_train_bal)}")
except Exception as e:
    print(f"Erro ao aplicar SMOTE: {e}. Usando dados originais.")
    X_train_bal, y_train_bal = X_train, y_train

# Otimização com Optuna 

def objective(trial):
    # Hiperparâmetros sugeridos pelo Optuna
    
    # Estrutura da Rede KAN
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64])
    hidden_sizes = tuple([hidden_size] * n_layers)
    
    # Parâmetros Específicos do KAN (Math Magic)
    grid_size = trial.suggest_int('grid_size', 3, 8)
    spline_order = trial.suggest_int('spline_order', 2, 3)
    
    # Treinamento
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    model = KANClassifier(
        hidden_sizes=hidden_sizes,
        grid_size=grid_size,
        spline_order=spline_order,
        learning_rate=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=20, 
        device=DEVICE
    )
    
    # Treina no conjunto balanceado
    model.fit(X_train_bal, y_train_bal)
    
    # Avalia na validação (NÃO balanceada, dados reais)
    probs_val = model.predict_proba(X_val)[:, 1]
    
    # Métrica alvo: AUROC (Area Under ROC Curve) - Robustez geral
    try:
        score = roc_auc_score(y_val, probs_val)
    except ValueError:
        score = 0.5 
        
    return score

print("\nIniciando busca de hiperparâmetros com Optuna...")
optuna.logging.set_verbosity(optuna.logging.WARNING) # Menos texto na tela
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS)

print("\n Melhores parâmetros encontrados:")
print(study.best_params)

# Treinamento Final e Teste 
print("\n Treinando modelo final KAN com os melhores parâmetros...")

best_params = study.best_params
# Reconstroi a tupla hidden_sizes a partir dos params soltos
best_hidden_sizes = tuple([best_params['hidden_size']] * best_params['n_layers'])

final_model = KANClassifier(
    hidden_sizes=best_hidden_sizes,
    grid_size=best_params['grid_size'],
    spline_order=best_params['spline_order'],
    learning_rate=best_params['lr'],
    weight_decay=best_params['weight_decay'],
    batch_size=best_params['batch_size'],
    max_epochs=50, 
    device=DEVICE
)

final_model.fit(X_train_bal, y_train_bal)

# Avaliação Final no Teste
print("\n Resultados no Conjunto de TESTE (Final):")
probs_test = final_model.predict_proba(X_test)[:, 1]
preds_test = (probs_test >= 0.5).astype(int)

print(classification_report(y_test, preds_test))
print(f"AUROC Final: {roc_auc_score(y_test, probs_test):.4f}")