import numpy as np 
from torchvision import datasets 

REGISTRY = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
    "kmnist": datasets.KMNIST
}

class ImageDataset: 
    def __init__(self, cfg, root = 'data', seed = 67): 
        raw = REGISTRY[cfg.name](root, train = True, download = True)
        X = raw.data.numpy().reshape(len(raw.data), -1).astype(np.float32) / 255.0
        Y = np.asarray(raw.targets).astype(int)
        self.image_shape = tuple(raw.data.shape[1:])
        self._rng = np.random.default_rng(seed)

        if cfg.binarize: 
            X = (X > 0.5).astype(np.float32)

        idx = self._rng.permutation(len(X)) 
        n_val = int(len(X) * cfg.val_fraction)
        self.val, self.train = X[idx[:n_val]], X[idx[n_val:]]
        self.y_val, self.y_train = Y[idx[:n_val]], Y[idx[n_val:]]
        self.n_features = X.shape[1]

    def batches(self, batch_size):
        idx = self._rng.permutation(len(self.train))

        for i in range(0, len(idx) - batch_size + 1, batch_size): 
            yield self.train[idx[i:i + batch_size]]  