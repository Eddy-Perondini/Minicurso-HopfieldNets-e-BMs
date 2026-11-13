from dataclasses import dataclass

@dataclass 
class DataConfig:
    name: str = "mnist"
    binarize: bool = True 
    val_fraction: float = 0.1 

@dataclass
class ModelConfig: 
    n_hidden: int = 500 
    seed: int = 67 

@dataclass 
class TrainingConfig:
    epochs: int = 30 
    batch_size: int = 128
    lr: float = 0.05
    k: int = 1 #passos de Gibbs ==> CD-k 
    persistent: bool = True # True: PCD, False: CD-k 
    momentum: float = 0.5
    final_momentum: float = 0.5
    weight_decay: float = 1e-3 

