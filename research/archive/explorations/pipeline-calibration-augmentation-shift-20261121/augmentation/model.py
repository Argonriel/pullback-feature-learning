"""Small fixed CNN and deterministic runtime, without torchvision."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn


def runtime():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64*7*7,128), nn.ReLU(), nn.Linear(128,10))

    def forward(self, x):
        return self.head(self.features(x))


def load_model(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    net = CNN(); net.load_state_dict(payload['state_dict']); net.to(device); net.eval()
    return net


@torch.inference_mode()
def predict_chunks(net, x, batch=512):
    return torch.cat([net(x[j:j+batch]) for j in range(0,len(x),batch)])
