import numpy as np 
import matplotlib.pyplot as plt 

def _grid(imgs, shape, ncols): 
    n = len(imgs); nrows = int(np.ceil(n / ncols))

    canvas = np.zeros((nrows * shape[0], ncols * shape[1]))

    for k, im in enumerate(imgs): 
        r,c = divmod(k, ncols)
        canvas[r*shape[0]:(r+1)*shape[0], c*shape[1]:(c+1)*shape[1]] = im.reshape(shape)

    return canvas 

def save_grid(imgs, shape, path, ncols=10): 
    plt.figure(figsize=(8, 8)); plt.imshow(_grid(imgs, shape, ncols), cmap="gray")
    plt.axis("off"); plt.savefig(path, bbox_inches="tight", dpi=150); plt.close()

def save_filters(rbm, shape, path, n=100):
    save_grid(rbm.W.T[:n], shape, path)    #cada coluna de W é um "detector" de features 
