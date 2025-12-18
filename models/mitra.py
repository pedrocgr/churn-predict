import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

# Arquitetura MITRA (Transformer-based for Tabular Data) 
# Implementação baseada em Foundation Models modernos para tabelas.
# Utiliza mecanismos de atenção (Self-Attention) para capturar relações complexas.

class MitraModule(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        
        # Feature Embedding:
        # Projeta as features originais para a dimensão do modelo
        self.embedding = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        # Transformer Encoder:
        # Aprende correlações globais entre as variáveis usando Multi-Head Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
            activation="gelu" # GELU é o padrão atual (usado no BERT/GPT)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction Head:
        # Classificador final
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid() # Probabilidade 0-1
        )

    def forward(self, x):
        # x shape: [batch_size, num_features]
        
        # Projeção
        x = self.embedding(x) # [batch_size, d_model]
        
        # O Transformer espera uma sequência [batch, seq_len, features].
        # Tratamos o cliente como uma sequência de tamanho 1.
        x = x.unsqueeze(1) # [batch_size, 1, d_model]
        
        # Passa pelo Transformer
        x = self.transformer_encoder(x)
        
        # Remove a dimensão da sequência
        x = x.squeeze(1) # [batch_size, d_model]
        
        return self.head(x)

# Wrapper Scikit-Learn 
# Permite usar o modelo com Optuna e pipelines do Sklearn
class MitraClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1,
                 learning_rate=0.001, max_epochs=50, batch_size=64, device='cpu'):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.losses_ = []

    def fit(self, X, y):
        # Converte dados para Tensores PyTorch
        X_tensor = torch.FloatTensor(X if isinstance(X, np.ndarray) else X.values).to(self.device)
        y_tensor = torch.FloatTensor(y if isinstance(y, np.ndarray) else y.values).reshape(-1, 1).to(self.device)

        input_size = X.shape[1]
        
        # Instancia o modelo
        self.model = MitraModule(
            input_size, 
            self.d_model, 
            self.nhead, 
            self.num_layers, 
            self.dim_feedforward, 
            self.dropout
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        criterion = nn.BCELoss()
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.max_epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            self.losses_.append(epoch_loss / len(loader))
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ['model'])
        self.model.eval()
        X_tensor = torch.FloatTensor(X if isinstance(X, np.ndarray) else X.values).to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()
        return np.hstack([1-preds, preds])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)