from model.neural_network import MSINTModel
from train import train_model
from loader import get_data_loaders
from evaluate import evaluate_model, plot_confusion_matrix
import torch.nn as nn
import torch.optim as optim
import os
import torch

train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)

model = MSINTModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, num_epochs=50)

print("\nAvaliação no conjunto de validação:")
evaluate_model(model, val_loader, criterion)

print("\nAvaliação final no conjunto de teste:")
evaluate_model(model, test_loader, criterion)

plot_confusion_matrix(model, test_loader)

model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'mnist_model.pt')
torch.save(model.state_dict(), model_path)

print(f"Modelo salvo em {model_path}")