import numpy as np 

class Evaluator: 
    def __init__(self, rbm): 
        self.rbm = rbm 

    def reconstruction_error(self, v): 
        pv = self.rbm.visible_probs(self.rbm.hidden_probs(v))
        return float(((v-pv)**2).mean()) #MSE entre v e p(v|h) 

    def pseudo_loglik(self, v): 

        rbm = self.rbm 
        i = rbm.rng.integers(0, v.shape[1], size = len(v)) 
        v_flip = v.copy()
        rows = np.arange(len(v))
        v_flip[rows, i] = 1 - v_flip[rows, i] 
        d = rbm.free_energy(v_flip) - rbm.free_energy(v) 
        return float(v.shape[1] * np.log(1.0 /(1.0 + np.exp(-d))).mean()) 

    def run(self, dataset): 
        tr = dataset.train[:5000] #subamostra de treino para acelerar a avaliação 
        return {
            "reconstruction_error": self.reconstruction_error(tr), 
            "pseudo_loglik": self.pseudo_loglik(tr),
            "FE_gap": float(self.rbm.free_energy(dataset.val).mean() - self.rbm.free_energy(tr).mean()) 
        }

    def dead_units(self, threshold = 0.5): 
        '''conta unidades ocultas cujo vetor de pesos é bem menor que o comum'''

        norms = np.linalg.norm(self.rbm.W, axis = 0)
        dead = norms < threshold * np.median(norms)

        return int(dead.sum()), len(norms)
    