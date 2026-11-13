import numpy as np 

class Trainer: 
    def __init__(self, rbm, cfg, evaluator = None):
        self.rbm = rbm 
        self.cfg = cfg
        self.evaluator = evaluator 

        self.vW = np.zeros_like(rbm.W) #inicializa o momento do peso
        self.vb = np.zeros_like(rbm.b) #inicializa o momento do viés visível 
        self.vc = np.zeros_like(rbm.c) #inicializa o momento do viés oculto

        self._chain = None #estado de cadeia persistente 

        self.history = []

    def _negative_phase(self, v0): 
        rbm = self.rbm 
        v = self._chain if (self.cfg.persistent and self._chain is not None) else v0 #inicializa a cadeia com a amostra persistente ou com a amostra atual

        for _ in range(self.cfg.k): 
            v, pv  = rbm.gibbs_step(v) #realiza k passos de Gibbs 

        if self.cfg.persistent:
            self._chain = v #atualiza a amostra persistente 

        return pv, rbm.hidden_probs(v) #retorna p(v|h) e p(h|v) 

    def _step(self, v0, momentum): 
        rbm = self.rbm  
        c = self.cfg 

        n = len(v0) 
        ph0 = rbm.hidden_probs(v0) #calcula p(h|v0), fase positiva 
        vk, phk = self._negative_phase(v0) #fase negativa 

        dW = (v0.T @ ph0 - vk.T @ phk) / n - c.weight_decay * rbm.W #gradiente do peso 
        db = (v0 - vk).mean(axis = 0) #gradiente do viés visível 
        dc = (ph0 - phk).mean(axis = 0) #gradiente do viés oculto 

        self.vW = momentum * self.vW + c.lr * dW #atualiza o momento do peso 
        self.vb = momentum * self.vb + c.lr * db #atualiza o momento do viés visível
        self.vc = momentum * self.vc + c.lr * dc #atualiza o momento do viés oculto  

        rbm.W += self.vW; rbm.b += self.vb; rbm.c += self.vc #atualiza os parâmetros do RBM 


    def fit(self, dataset): 
        c = self.cfg 

        for epoch in range(c.epochs): 
            momentum = c.momentum if epoch < 5 else c.final_momentum #momento baixo no inicio e alto no final do treinamento 

            for batch in dataset.batches(c.batch_size): 
                self._step(batch, momentum) 

            if self.evaluator: 
                metrics = self.evaluator.run(dataset)
                self.history.append(metrics)

                print(f"epoch {epoch+1:3d} | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        return self.history 
