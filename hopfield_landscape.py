'''
Seja uma rede de Hopfield com estados s {+1, -1} 

e 

E(s) = -1/2 s^T W s + theta^T s

com o vetor de pesos sinápticos W simétricos e diagonal nula. 

O espaço de estados é um hipercubo de dimensão N, então temos quatro recortes complementares: 

  1. flip_path / plot_flip_path : energia ao longo de um caminho de bit-flips entre dois
                                  estados (mostra a barreira entre dois atratores)
  2. basin_map                  : mapa 2D das bacias de atração num plano escolhido
  3. plot_point_cloud           : para N pequeno (<= 18), todos os 2^N estados com sua energia
  4. find_attractors            : lista os pontos fixos alcançados a partir de inícios aleatórios
'''

#Importações 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch 

PATH = '/home/EddyPerondini/Documentos/Mestrado/Minicurso/imagens'

#Funções -- Núcleo da rede 

def energy(W, s, theta = None): 
    s = np.asarray(s, dtype = float)
    e = - 0.5 * np.einsum("...i,ij,...j->...", s, W, s)

    if theta is not None: 
        e = e + s @ theta
    return e

def relax(W, s0, theta = None, max_sweeps = 100, rng = None): 
    rng = rng or np.random.default_rng(0)
    s = np.array(s0, dtype=float)
    n = len(s)
    th = np.zeros(n) if theta is None else theta 

    for _ in range(max_sweeps): 
        changed = False 
        
        for i in rng.permutation(n): 
            new = 1.0 if (W[i] @ s - th[i]) >= 0 else -1.0 
            
            if new != s[i]:
                s[i] = new 
                changed = True 

        if not changed: 
            break

    return s

def find_attractors(W, theta = None, n_starts = 300, seed = 67): 
    rng = np.random.default_rng(seed)
    n = W.shape[0]

    finals = np.array([relax(W, rng.choice([-1.0,1.0], n), theta, rng = rng)
                       for _ in range(n_starts)])

    uniq, counts = np.unique(finals, axis = 0, return_counts = True)
    
    order = np.argsort(-counts)

    return uniq[order], counts[order]

#Funções -- Energia ao longo de um caminho

def flip_path(energy_fn, a, b, n_orders = 30, seed = 67): 
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype = float)
    b = np.asarray(b, dtype = float)
    diff = np.flatnonzero(a != b)
    paths = []

    for _ in range(n_orders): 
        S = np.tile(a, (len(diff) + 1, 1))

        for k, i in enumerate(rng.permutation(diff)): 
            S[k+1:, i] = b[i]

        paths.append(energy_fn(S))

    return np.array(paths)

def plot_flip_path(energy_fn, a, b, labels = ("A","B"), n_orders = 30, title = None, path = PATH): 
    P = flip_path(energy_fn, a, b, n_orders) 
    x = np.linspace(0, 1, P.shape[1])
    best = P.max(axis = 1).argmin()
    barrier = P[best].max() - max(P[best,0], P[best,-1])

    fig, ax = plt.subplots(figsize = (7,4))
    ax.plot(x, P.T, color = 'gray', alpha = 0.25, lw = 0.8)
    ax.plot(x, P.mean(axis = 0), color = 'C0', lw = 2, label = "média dos caminhos")
    ax.plot(x, P[best], color = 'C3', lw = 2, label = "caminho de menor barreira")

    ax.set_xlabel(f"fração de bits trocados ({labels[0]} -> {labels[1]})")
    ax.set_ylabel("energia")
    ax.set_title(title or f"Barreira = {barrier:.2f}")
    ax.legend()

    if path: 
        fig.savefig(f'{path}/plot_flip_path.pdf', dpi=150)
    return fig, barrier

#Funções -- Mapa de Bacias 2D

def basin_map(W, attractors, theta=None, res = 41, margin = 0.4, top = 6, seed = 67, path = PATH): 
    rng = np.random.default_rng(seed)
    vals, vecs = np.linalg.eigh(W)
    u1, u2 = vecs[:,-1], vecs[:,-2]
    U = np.stack([u1,u2], axis = 1)

    proj = attractors @ U
    lo = np.minimum(proj.min(0), 0)
    hi = np.maximum(proj.max(0), 0)
    pad = margin * (hi - lo)

    xs = np.linspace(lo[0] - pad[0], hi[0] + pad[0], res)
    ys = np.linspace(lo[1] - pad[1], hi[1] + pad[1], res)

    ids = np.full((res, res), top)
    efinal = np.zeros((res, res))

    for j,y in enumerate(ys): 
        for i, x in enumerate(xs): 
            s = relax(W, x * u1 + y * u2, rng = rng)
            efinal[j,i] = energy(W, s, theta)
            match = np.flatnonzero((attractors[:top] == s).all(axis = 1))

            if len(match): 
                ids[j,i] = match[0]

    k = min(top, len(attractors))
    colors = list(plt.cm.tab10(np.arange(k))) + [(0.8, 0.8, 0.8, 1.0)]
    cmap = ListedColormap(colors[:k] + [colors[-1]])
    ids = np.minimum(ids, k)
    extent = (xs[0], xs[-1], ys[0], ys[-1])

    fig,axes = plt.subplots(1, 2, figsize = (12,5))
    axes[0].imshow(ids, origin = 'lower', extent = extent, cmap = cmap, vmin = 0, vmax = k, aspect = 'auto')
    handles = [Patch(color=colors[c],
               label=f"atrator {c} (E = {energy(W, attractors[c], theta):.1f})")
               for c in range(k)]
    handles.append(Patch(color = colors[-1], label = 'outros'))
    axes[0].legend(handles=handles, fontsize = 10, loc = 'upper right')
    axes[0].set_title("Bacias de Atração")

    im = axes[1].imshow(efinal, origin = 'lower', extent = extent, cmap = 'viridis', aspect = 'auto')
    fig.colorbar(im, ax=axes[1], label = 'energia do ponto fixo')
    axes[1].set_title("Energia do estado final")

    for ax in axes: 
        ax.scatter(proj[:top,0], proj[:top, 1], c = 'white', edgecolor = 'black', marker = '*', s = 40, zorder = 3)
        ax.set_xlabel('autovetor 1 de W')
        ax.set_ylabel('autovetor 2 de W')
    fig.tight_layout()

    if path: 
        fig.savefig(f'{path}/basin_map.pdf', dpi = 150)

    return fig

def enumerate_states(n): 
    idx = np.arange(2 ** n)
    return 2.0 * ((idx[:, None] >> np.arange(n)) & 1) - 1.0

def plot_point_cloud(W, theta = None, path = PATH): 
    n = W.shape[0]

    assert n <= 18, "enumeração exata só para N <= 18"
    S = enumerate_states(n)
    E = energy(W, S, theta)

    idx = np.arange(len(S)) 
    is_min = np.ones(len(S), dtype = bool)
    for i in range(n): 
        is_min &= E <= E[idx ^ (1 << i)]

    _, vecs = np.linalg.eigh(W)
    X,Y = S @ vecs[:,-1], S @ vecs[:,-2]

    fig = plt.figure(figsize = (8,6))
    ax = fig.add_subplot(projection = '3d')
    sc = ax.scatter(X, Y, E, c = E, cmap = 'viridis', s = 4, alpha = 0.5)
    ax.scatter(X[is_min], Y[is_min], E[is_min], c = 'red', marker = '*', s = 120, label = 'mínimos locais')
    fig.colorbar(sc, ax=ax, shrink = 0.6, label = 'energia')
    ax.set_xlabel("autovetor 1 de W")
    ax.set_ylabel("autovetor 2 de W")
    ax.set_zlabel("energia")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(f'{path}/plot_point_cloud.pdf', dpi=150)
    print(f"{is_min.sum()} mínimos locais em {len(S)} estados")
    return fig

#Main

def hebbian(patterns): 
    P = np.asarray(patterns, dtype = float)
    W = P.T @ P / P.shape[1]
    np.fill_diagonal(W, 0.0)
    return W 

if __name__ == "__main__": 
    rng = np.random.default_rng(0)

    pats = rng.choice([-1.0, 1.0], size=(4, 100))
    W = hebbian(pats)
    att, cnt = find_attractors(W)
    print(f"{len(att)} atratores distintos; bacias: {cnt[:8]}")

    plot_flip_path(lambda S: energy(W,S), pats[0], pats[1], labels = ("Padrão 0", "Padrão 1"))

    basin_map(W, att)

    W_small = hebbian(rng.choice([-1.0,1.0], size = (3, 14)))
    att_small, cnt_small = find_attractors(W_small)
    print(f"{len(att_small)} atratores distintos; bacias: {cnt_small[:]}")
    plot_point_cloud(W_small)
    plt.show()