from .base_dataset import Dataset

# from .mock_1d import Mock1D
# from .mock_2d import Mock2D
from .mock_unbalanced_cluster import MockUnbalancedCluster
from .mock_from_model import MockFromModel
from .mock_mismatch import MockMismatched

from .data_uci import UCI, UCI_SETS_CLASSIFICATION, UCI_SETS_REGRESSION
from .emnist import EMNIST
from .cifar import CIFAR
from .svhn import SVHN