from config import DataConfig, ModelConfig, TrainingConfig
from data import ImageDataset
from model import RBM
from trainer import Trainer
from evaluation import Evaluator
from visualization import save_grid, save_filters

def main(): 
    dcfg = DataConfig()
    mcfg = ModelConfig()
    tcfg = TrainingConfig() 
    ds = ImageDataset(dcfg) 
    rbm = RBM(ds.n_features, mcfg.n_hidden, seed = mcfg.seed)
    trainer = Trainer(rbm, tcfg, Evaluator(rbm))
    trainer.fit(ds, checkpoint_path="checkpoints/rbm.pkl", every=1)
    rbm.save("checkpoints/rbm_final.npz")

    
    n_dead, n_total = Evaluator(rbm).dead_units()
    print(f"{n_dead} unidades mortas de {n_total}")

    save_filters(rbm, ds.image_shape, 'filters.png') 

    samples = rbm.generate(100, n_steps = 1000, v_init = ds.val[:100])

    save_grid(samples, ds.image_shape, 'samples.png')

if __name__ == "__main__": 
    main()