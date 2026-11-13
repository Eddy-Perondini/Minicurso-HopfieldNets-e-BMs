# Minicurso: Introdução à Redes Neurais de Hopfield e Máquinas de Boltzmann

Implementação didática de uma **Restricted Boltzmann Machine (RBM) Bernoulli-Bernoulli** em NumPy puro, treinada com **Contrastive Divergence (CD-k)** ou **Persistent CD (PCD)** em benchmarks de imagens (MNIST, Fashion-MNIST, KMNIST).

O código é organizado em módulos com responsabilidades claras, sem fragmentar em excesso: cada arquivo tem um conjunto de funções com um papel bem definido.

## Instalação

```bash
pip install numpy matplotlib torchvision
```

O `torchvision` é usado **apenas para baixar os datasets**. O modelo e o treino não dependem de PyTorch.

## Estrutura do projeto

```
rbm_project/
├── config.py          # dataclasses com hiperparâmetros
├── data.py            # ImageDataset: carrega, binariza, divide, gera batches
├── model.py           # RBM: parâmetros, probabilidades, amostragem, energia livre, save/load
├── trainer.py         # Trainer: CD-k / PCD, momentum, weight decay, checkpoint
├── evaluation.py      # Evaluator: erro de reconstrução, pseudo-LL, gap de energia livre
├── visualization.py   # grades de amostras e filtros aprendidos
└── train.py           # ponto de entrada que monta tudo
```

| Módulo | Responsabilidade | Não sabe nada sobre |
|---|---|---|
| `model.py` | Modelo probabilístico: calcula e amostra | treino, dados |
| `trainer.py` | Algoritmo de aprendizado | como o dataset é carregado |
| `data.py` | Carregar, binarizar, dividir, gerar batches | o modelo |
| `evaluation.py` | Métricas de qualidade | como o treino funciona |
| `visualization.py` | Gerar imagens (funções puras) | todos os outros |

## Como os módulos se conectam

`train.py` é a **raiz de composição**: ele instancia os objetos e entrega cada um a quem precisa (injeção de dependência). Os demais módulos não se importam entre si.

```
train.py ──► config
   ├──► data           (ImageDataset)
   ├──► model          (RBM)            não conhece trainer nem data
   ├──► trainer        (usa RBM e config; recebe o dataset em fit())
   ├──► evaluation     (usa RBM; recebe o dataset em run())
   └──► visualization
```

Durante o treino, o `Trainer` chama `evaluator.run(dataset)` ao fim de cada época.

A regra principal: **o modelo só sabe calcular e amostrar; quem sabe treinar é o `Trainer`.** Assim, trocar CD por PCD, ou o otimizador, não exige mexer no modelo.

## Uso rápido

```bash
python train.py
```

Isso treina a RBM e gera:

- `filters.png`: filtros aprendidos (cada coluna de `W` é um detector de feature)
- `samples.png`: amostras geradas por uma cadeia de Gibbs longa

Para gerar amostras a partir de um modelo já treinado, sem treinar de novo:

```python
from data import ImageDataset
from config import DataConfig
from model import RBM
from visualization import save_grid

ds = ImageDataset(DataConfig())
rbm = RBM.load("checkpoints/rbm_final.npz")
samples = rbm.generate(100, n_steps=1000, v_init=ds.val[:100])
save_grid(samples, ds.image_shape, "samples.png")
```

## Configuração

Todos os hiperparâmetros ficam em `config.py`.

```python
@dataclass
class DataConfig:
    name: str = "mnist"          # mnist | fashion_mnist | kmnist
    binarize: bool = True
    val_fraction: float = 0.1

@dataclass
class ModelConfig:
    n_hidden: int = 256
    seed: int = 0

@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 100
    lr: float = 0.05
    k: int = 1                   # passos de Gibbs na fase negativa
    persistent: bool = False     # True = PCD
    momentum: float = 0.5
    final_momentum: float = 0.5  
    weight_decay: float = 1e-4
```

Configuração que funcionou bem para **gerar** no MNIST (PCD estável):

```python
epochs = 50
lr = 0.01
k = 1
persistent = True
momentum = 0.5
final_momentum = 0.5      
weight_decay = 1e-3
```

## Checkpoints

O treino salva o estado completo e pode ser retomado.

```python
trainer = Trainer(rbm, tcfg, Evaluator(rbm))
trainer.fit(ds, checkpoint_path="checkpoints/rbm.pkl", every=1)
rbm.save("checkpoints/rbm_final.npz")
```

- **`rbm.save` / `RBM.load`** (`.npz`): só os parâmetros `W`, `b`, `c`. Serve para gerar amostras depois.
- **`Trainer.save_checkpoint` / `load_checkpoint`** (`.pkl`): parâmetros, velocidades do momentum, cadeia persistente, histórico, época atual e estado do gerador aleatório. Serve para retomar o treino.

Comportamento:

- Se o arquivo de checkpoint existir, o `fit` retoma da última época salva.
- Para treinar mais, aumente `epochs` e rode de novo com o mesmo caminho.
- A escrita é atômica (arquivo temporário + `os.replace`), então uma queda no meio não corrompe o checkpoint.

Cuidados:

1. **Mudou hiperparâmetros do modelo ou do treino** (`n_hidden`, `lr`, `persistent`, dataset)? Use um caminho novo ou apague o checkpoint. Por exemplo: `checkpoints/mnist_h256_pcd.pkl`.
2. **Mantenha a mesma `seed` no `ImageDataset`**, para a divisão treino/validação ser idêntica ao retomar.
3. Só carregue arquivos `.pkl` que você mesmo gerou (`pickle` executa código ao carregar).

## Métricas

Calculadas pelo `Evaluator` ao fim de cada época:

| Métrica | Descrição |
|---|---|---|
| `reconstruction_error` | Erro quadrático médio por pixel entre `v` e `p(v\|h)` |
| `pseudo_loglik` | Pseudo-log-verossimilhança (inverte 1 bit aleatório por exemplo) |
| `FE_gap` | Energia livre média da validação menos a do treino | 

Observação: Uma RBM pode reconstruir bem e gerar mal. **Julgue pelas amostras.**

## Referências

- Hinton, G. E. *A Practical Guide to Training Restricted Boltzmann Machines* (2010).
- Tieleman, T. *Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient* (2008), artigo do PCD.
- Hinton, G. E. *Training Products of Experts by Minimizing Contrastive Divergence* (2002), artigo do CD.
