import numpy as np 

class RBM: 
    def __init__(self, n_visible, n_hidden, seed = 67): 
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0, 0.01, size = (n_visible, n_hidden))
        self.b = np.zeros(n_visible) #vies visivel da rede 
        self.c = np.zeros(n_hidden) #vies oculto da rede 

    @staticmethod
    def _sigmoid(x): 
        return 1.0/(1.0 + np.exp(-x)) #função de ativação sigmoide hardcodada 

    def hidden_probs(self,v):
        return self._sigmoid(v @ self.W + self.c) #p(h|v) = sigmoid(vW + c) 

    def visible_probs(self,h): 
        return self._sigmoid(h @ self.W.T + self.b) #p(v|h) = sigmoid(hW^T + b) 

    def sample(self, p): 
        return (self.rng.random(p.shape) < p).astype(np.float32) #amostragem de Bernoulli

    def gibbs_step(self, v): 

        '''
        Retorna (v_amostrado, prob_visivel) v -> h -> v' 
        '''

        h = self.sample(self.hidden_probs(v)) #amostra h a partir de p(h|v)
        pv = self.visible_probs(h) #calcula p(v|h)

        return self.sample(pv), pv 

    def free_energy(self, v): 

        '''
        Retorna a energia livre F(v) = -b^Tv - sum(log(1 + exp(c + W^Tv)))
        '''

        return -(v @ self.b) - np.sum(np.log(1 + np.exp(self.c + v @ self.W)), axis = 1)

    def generate(self, n_samples, n_steps = 1000, v_init = None): 
        v = v_init if v_init is not None else self.sample(np.full((n_samples, self.W.shape[0]), 0.5)) #inicializa v com amostras de Bernoulli p = 0.5 

        for _ in range(n_steps): 
            v, pv = self.gibbs_step(v) #realiza passos de Gibbs

        return pv 

    def save(self, path): 
        np.savez(path, W = self.W, b = self.b, c = self.c) #salva os parâmetros do RBM em um arquivo .npz 

    @classmethod 
    def load(cls, path):
        d = np.load(path) #carrega os parâmetros do RBM de um arquivo .npz 
        rbm = cls(d["W"].shape[0], d["W"].shape[1], seed = 67)
        rbm.W, rbm.b, rbm.c = d["W"], d["b"], d["c"]
        return rbm 