import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

# Camada KAN 
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(
                (self.scale_spline * noise).permute(2, 1, 0)
                if self.spline_weight.shape == noise.permute(2,1,0).shape else 
                torch.randn_like(self.spline_weight) * 0.1
            )

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def forward(self, x):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

# Modelo KAN 
class KAN(nn.Module):
    def __init__(self, input_size, hidden_sizes, grid_size=5, spline_order=3, output_size=1):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        in_dim = input_size
        
        for h_dim in hidden_sizes:
            self.layers.append(
                KANLinear(in_dim, h_dim, grid_size=grid_size, spline_order=spline_order)
            )
            in_dim = h_dim
            
        self.final_layer = nn.Linear(in_dim, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(self.final_layer(x))

# Wrapper Scikit-Learn 
class KANClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_sizes=(64, 32), grid_size=5, spline_order=3, 
                 learning_rate=0.01, weight_decay=1e-4, max_epochs=100, batch_size=64, device='cpu'):
        self.hidden_sizes = hidden_sizes
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.losses_ = []

    def fit(self, X, y):
        # Para garantir tensores float
        X_tensor = torch.FloatTensor(X if isinstance(X, np.ndarray) else X.values).to(self.device)
        y_tensor = torch.FloatTensor(y if isinstance(y, np.ndarray) else y.values).reshape(-1, 1).to(self.device)

        input_size = X.shape[1]
        self.model = KAN(
            input_size, 
            self.hidden_sizes, 
            grid_size=self.grid_size, 
            spline_order=self.spline_order
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
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