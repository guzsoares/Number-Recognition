import torch
from model.neural_network import MSINTModel

def load_model(model_path="models/mnist_model.pt"):
    model = MSINTModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        _, predicted = output.max(1)
    return predicted.item()