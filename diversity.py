import numpy as np

from config import DataConfig
from data import ImageDataset
from model import RBM
from evaluation import Evaluator
from classifier import SoftmaxClassifier
import pickle

N_SAMPLES = 1000
STEPS = (50, 200)


def main():
    ds = ImageDataset(DataConfig())

    clf = SoftmaxClassifier(ds.n_features).fit(ds.train, ds.y_train)
    print(f"acurácia do classificador (validação): {clf.accuracy(ds.val, ds.y_val):.3f}\n")

    rbm = RBM.load("checkpoints/rbm_final.npz")
    ev = Evaluator(rbm)

    rng = np.random.default_rng(1)
    init = ds.train[rng.choice(len(ds.train), N_SAMPLES, replace=False)]

    real = np.bincount(ds.y_val, minlength=10) / len(ds.y_val)
    print("classe       ", "  ".join(f"{c:4d}" for c in range(10)))
    print("dados reais  ", "  ".join(f"{p:4.2f}" for p in real))

    for steps in STEPS:
        p, h = ev.class_histogram(clf, N_SAMPLES, steps, v_init=init)
        print(f"{steps:5d} passos ", "  ".join(f"{x:4.2f}" for x in p), f"| entropia={h:.3f}")

    with open("checkpoints/rbm.pkl", "rb") as f:
        chain = pickle.load(f)["chain"]
    pred = clf.predict((chain > 0.5).astype(np.float32))
    print("cadeias do treino:", np.round(np.bincount(pred, minlength=10) / len(pred), 2))


if __name__ == "__main__":
    main()