# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


## 🚩🚩🚩 MODEL DEFINITIONS 🚩🚩🚩

class BirdCLEFModel(nn.Module):
    
    def __init__(self, input_size, hidden_size=128, num_classes=182, dropout=0.3):
        super().__init__()
        self.model_name = "BirdCLEFModel"
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size * 2, 64, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64 * 2, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x, lengths=None):
        # x shape: (batch, seq_len, features)
        # LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Take the last output for each sequence
        if lengths is not None:
            # Use the actual sequence lengths
            batch_size = x.size(0)
            x = torch.stack([x[i, lengths[i]-1, :] for i in range(batch_size)])
        else:
            # Use the last timestep
            x = x[:, -1, :]
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


# Pure CNN classifier for spectrogram input (3 conv blocks for speed)
class BirdClefCNNModel(nn.Module):
    def __init__(self, n_mels=None, time_steps=None, num_classes=None, dropout=0.3):
        super(BirdClefCNNModel, self).__init__()
        self.model_name = "BirdClefCNNModel"
        n_mels = n_mels
        if n_mels is None:
            raise ValueError("n_mels must be provided either in h dict or as parameter")
        if time_steps is None:
            raise ValueError("time_steps must be provided as parameter")
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # Validate input size for pooling
        min_time_steps = 2 ** 3  # 8 for 3 poolings
        if time_steps < min_time_steps:
            raise ValueError(f"time_steps must be at least {min_time_steps} for this model, got {time_steps}")

        # Global average pooling + simple classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, time_steps, n_mels)
        # Reshape to (batch, 1, n_mels, time_steps)
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        # Global average pooling: (batch, 128, h, w) -> (batch, 128, 1, 1)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # (batch, 128)
        x = self.fc(x)
        return x



# Define the model as a custom nn.Module to handle LSTM output
class CryLSTMClassifier(nn.Module):
    def __init__(self, h, n_mels=None, num_classes=None):
        super(CryLSTMClassifier, self).__init__()
        self.h = h
        self.model_name = "CryLSTMClassifier"
        n_mels = n_mels or h["n_mels"]
        self.lstm1 = nn.LSTM(input_size=n_mels, hidden_size=128, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.2)
        self.linear1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(128, num_classes)
        

    def forward(self, x):
        # LSTM layers return output and (h_n, c_n), we only need the output
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # Take the output of the last time step for classification
        # If batch_first is True, x has shape (batch, seq_len, num_directions * hidden_size)
        # We need the output of the last time step: x[:, -1, :]
        x = self.linear1(x[:, -1, :])
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.linear2(x)
        return x
    
# Define the model as a custom nn.Module to handle LSTM output
class CryLSTMClassifierBatchNorm(nn.Module):
    def __init__(self, h, n_mels=None, num_classes=None, dropout=0.3):
        super(CryLSTMClassifierBatchNorm, self).__init__()
        self.h = h
        self.model_name = "CryLSTMClassifierBatchNorm"
        n_mels = n_mels or h["n_mels"]
        self.lstm1 = nn.LSTM(input_size=n_mels, hidden_size=128, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(256)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(128, num_classes)
        

    def forward(self, x):
        # LSTM layers return output and (h_n, c_n), we only need the output
        x, _ = self.lstm1(x)

        # BatchNorm expects (batch, features, seq_len), so permute
        x = x.permute(0, 2, 1)
        x = self.bn1(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm2(x)
        x = x.permute(0, 2, 1)
        x = self.bn2(x)
        x = x.permute(0, 2, 1)

        # Take the output of the last time step for classification
        # If batch_first is True, x has shape (batch, seq_len, num_directions * hidden_size)
        # We need the output of the last time step: x[:, -1, :]
        x = self.linear1(x[:, -1, :])
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.linear2(x)
        return x
    
# Conv + LSTM classifier
class CryLSTMConvClassifier(nn.Module):
    def __init__(self, h, n_mels=None, num_classes=None, dropout=0.3):
        super(CryLSTMConvClassifier, self).__init__()
        self.h = h
        self.model_name = "CryLSTMConvClassifier"
        n_mels = n_mels or h["n_mels"]
        # Conv2d expects input (batch, channels, n_mels, time_steps)
        self.conv = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))
        self.bn_conv = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        
        # After pooling, n_mels and time_steps are halved
        # We'll flatten the conv output to (batch, time_steps, features) for LSTM
        self.lstm1 = nn.LSTM(input_size=(n_mels // 2) * 16, hidden_size=128, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(256)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear1 = nn.Linear(128, 128)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, time_steps, n_mels)
        # Reshape to (batch, 1, n_mels, time_steps)
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.conv(x)
        x = self.bn_conv(x)
        x = self.relu(x)
        x = self.pool(x)
        # x: (batch, channels=16, n_mels//2, time_steps//2)
        # Flatten for LSTM: (batch, time_steps//2, n_mels//2 * 16)
        b, c, n_mels_p, t_p = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t_p, c * n_mels_p)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)
        x = self.bn1(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x.permute(0, 2, 1)
        x = self.bn2(x)
        x = x.permute(0, 2, 1)
        x = self.linear1(x[:, -1, :])
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.linear2(x)
        return x

# Pure CNN classifier for spectrogram input (best pure CNN model)
class CryCNNClassifierComplex(nn.Module):
    def __init__(self, h, n_mels=None, time_steps=None, num_classes=None, dropout=0.3):
        super(CryCNNClassifierComplex, self).__init__()
        self.h = h
        self.model_name = "CryCNNClassifierComplex"
        n_mels = n_mels or h.get("n_mels")
        if n_mels is None:
            raise ValueError("n_mels must be provided either in h dict or as parameter")
        if time_steps is None:
            raise ValueError("time_steps must be provided as parameter")
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # Validate input size for pooling
        min_time_steps = 2 ** 4  # 16 for 4 poolings
        if time_steps < min_time_steps:
            raise ValueError(f"time_steps must be at least {min_time_steps} for this model, got {time_steps}")

        # Global average pooling + simple classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (batch, time_steps, n_mels)
        # Reshape to (batch, 1, n_mels, time_steps)
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        # Global average pooling: (batch, 256, h, w) -> (batch, 256, 1, 1)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # (batch, 256)
        x = self.fc(x)
        return x


class CryCNNClassifierComplex_InitializedWeights(nn.Module):
    def __init__(self, h, n_mels=None, num_classes=None, dropout=0.3):
        super(CryCNNClassifierComplex_InitializedWeights, self).__init__()
        self.h = h
        self.model_name = "CNNClassifierComplex_InitializedWeights"
        n_mels = n_mels or h.get("n_mels")
        if n_mels is None:
            raise ValueError("n_mels must be provided either in h dict or as parameter")
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        self.fc1 = None  # Will be defined after seeing input shape
        self.fc2 = nn.Linear(128, num_classes)
        self._init_weights()

    def forward(self, x):
        # x: (batch, time_steps, n_mels)
        # Reshape to (batch, 1, n_mels, time_steps)
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x)
        if self.fc1 is None:
            print(f"[DEBUG] Flattened shape: {x.shape}")
            self.fc1 = nn.Linear(x.shape[1], 128).to(x.device)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                    
# Pure CNN classifier with 6 conv layers and ReLU
class CryCNNClassifierComplex_6Conv(nn.Module):
    def __init__(self, h, n_mels=None, num_classes=None, dropout=0.3):
        super(CryCNNClassifierComplex_6Conv, self).__init__()
        self.h = h
        self.model_name = "CryCNNClassifierComplex_6Conv"
        n_mels = n_mels or h.get("n_mels")
        if n_mels is None:
            raise ValueError("n_mels must be provided either in h dict or as parameter")
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        self.fc1 = None  # Will be defined after seeing input shape
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, time_steps, n_mels)
        # Reshape to (batch, 1, n_mels, time_steps)
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)  # Pool after 2nd conv
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)  # Pool after 4th conv
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)  # Pool after 6th conv
        x = self.flatten(x)
        if self.fc1 is None:
            print(f"[DEBUG] Flattened shape: {x.shape}")
            self.fc1 = nn.Linear(x.shape[1], 128).to(x.device)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Pure CNN classifier with 8 conv layers and ReLU
class CryCNNClassifierComplex_8Conv(nn.Module):
    def __init__(self, h, n_mels=None, num_classes=None, dropout=0.3):
        super(CryCNNClassifierComplex_8Conv, self).__init__()
        self.h = h
        self.model_name = "CryCNNClassifierComplex_8Conv"
        n_mels = n_mels or h.get("n_mels")
        if n_mels is None:
            raise ValueError("n_mels must be provided either in h dict or as parameter")
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1)
        self.bn8 = nn.BatchNorm2d(1024)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        self.fc1 = None  # Will be defined after seeing input shape
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, time_steps, n_mels)
        # Reshape to (batch, 1, n_mels, time_steps)
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)  # Pool after 2nd conv
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)  # Pool after 4th conv
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)  # Pool after 6th conv
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)  # Pool after 8th conv
        x = self.flatten(x)
        if self.fc1 is None:
            print(f"[DEBUG] Flattened shape: {x.shape}")
            self.fc1 = nn.Linear(x.shape[1], 128).to(x.device)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# CNN + LSTM classifier for spectrogram input (fixed: all layers in __init__)
class CryCNNLSTMClassifier(nn.Module):
    def __init__(self, h, n_mels=None, num_classes=None, dropout=0.3, lstm_hidden=128, lstm_layers=1):
        super().__init__()
        self.h = h
        self.model_name = "CryCNNLSTMClassifier"
        n_mels = n_mels or h.get("n_mels")
        if n_mels is None:
            raise ValueError("n_mels must be provided either in h dict or as parameter")
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        # Calculate the feature size after convolutions and pooling
        # Each pooling halves the freq and time dims (4 times)
        freq_after = n_mels // (2 ** 4)
        cnn_feat_size = 256 * freq_after
        self.lstm = nn.LSTM(input_size=cnn_feat_size, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
    def forward(self, x):
        # x: (batch, time_steps, n_mels)
        x = x.permute(0, 2, 1).unsqueeze(1)  # (batch, 1, n_mels, time_steps)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        # x: (batch, channels=256, freq, time)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch, time, channels, freq)
        x = x.view(b, t, c * f)  # (batch, time, features)
        lstm_out, _ = self.lstm(x)  # (batch, time, hidden*2)
        x = lstm_out[:, -1, :]  # Take last time step
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Utility: Save model and load model functions
def save_model(model, h, run_id=None):
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_name = h.get("model_name", "model")
    if run_id is None:
        from datetime import datetime
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(model_dir, f"{model_name}_{run_id}.pt")
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_class, path, h, *args, **kwargs):
    model = model_class(h, *args, **kwargs)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    print(f"Model loaded from {path}")
    return model

