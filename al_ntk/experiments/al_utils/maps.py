from al_ntk.dataset import MockFromModel, MockMismatched, UCI, EMNIST, CIFAR, SVHN
from al_ntk.al_algorithms import RandomMethod, NTKGPApproxMethod, GreedyCentreMethod, BADGEMethod, \
    MCDropoutBasedMethod, KMeansPlusPlusMethod, NTKGPEnsembleMethod, MLMOCMethod, MLMOCMethodAlt


al_map = {
    'random': RandomMethod,
    'max-cov': GreedyCentreMethod,
    'ntkgp': NTKGPApproxMethod,
    'ntkgp-ensemble': NTKGPEnsembleMethod,
    'mlmoc': MLMOCMethod,
    'mlmoc-old': MLMOCMethodAlt,
    'badge': BADGEMethod,
    'mc-dropout': MCDropoutBasedMethod,
    'kmeans-pp': KMeansPlusPlusMethod
}

data_map = {
    'test': MockFromModel,
    'mock-mismatch': MockMismatched,
    'emnist': EMNIST,
    'uci': UCI,
    'cifar': CIFAR,
    'svhn': SVHN,
}