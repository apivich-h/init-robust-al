from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import SGD, Adam
from al_ntk.model import MLPJax, MLPTorch, CNNJax, CNNTorch, WideResNetTorch, ResNetTorch, EnsembleModelTorch, EnsembleModelJax


model_map = {
    'mlp': {
        'jax': MLPJax, 
        'torch': MLPTorch,
    },
    'cnn': {
        'jax': CNNJax,
        'torch': CNNTorch,
    },
    'wrn': {
        'jax': None,
        'torch': WideResNetTorch,
    },
    'resnet': {
        'jax': None,
        'torch': ResNetTorch,
    },
    'ensemble': {
        'jax': EnsembleModelJax,
        'torch': EnsembleModelTorch
    },
}

loss_map = {
    'regression': MSELoss,
    'binary_classification': BCEWithLogitsLoss,
    'classification': CrossEntropyLoss,
}

optim_map = {
    'sgd': SGD,
    'adam': Adam,
}
