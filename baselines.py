import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
from tqdm.auto import tqdm
import dagu

import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
from scipy.special import expit as sigmoid
from tqdm.auto import tqdm
from numpy import linalg as la
#===================================#
#   Equal variance  CoLiDE-EV       #
#===================================#

class colide_ev:
    
    def __init__(self, dtype=np.float64, seed=0):
        super().__init__()
        np.random.seed(seed)
        self.dtype = dtype
            
    def _score(self, W, sigma):
        dif = self.Id - W 
        rhs = self.cov @ dif
        loss = ((0.5 * np.trace(dif.T @ rhs)) / sigma) + (0.5 * sigma * self.d)
        G_loss = -rhs / sigma
        return loss, G_loss

    def dagness(self, W, s=1):
        value, _ = self._h(W, s)
        return value
    
    def _h(self, W, s=1.0):
        M = s * self.Id - W * W
        h = - la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T 
        return h, G_h

    def _func(self, W, sigma, mu, s=1.0):
        score, _ = self._score(W, sigma)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h 
        return obj, score, h
    
    def _adam_update(self, grad, iter, beta_1, beta_2):
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - beta_1 ** iter)
        v_hat = self.opt_v / (1 - beta_2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad
    
    def minimize(self, W, sigma, mu, max_iter, s, lr, tol=1e-6, beta_1=0.99, beta_2=0.999, pbar=None):
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        
        for iter in range(1, max_iter+1):
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < -1e-6):
                if iter == 1 or s <= 0.9:
                    return W, sigma, False
                else:
                    W += lr * grad
                    lr *= .5
                    if lr <= 1e-16:
                        return W, sigma, True
                    W -= lr * grad
                    dif = self.Id - W 
                    rhs = self.cov @ dif
                    sigma = np.sqrt(np.trace(dif.T @ rhs) / (self.d))
                    M = sla.inv(s * self.Id - W * W) + 1e-16
            
            G_score = -mu * self.cov @ (self.Id - W) / sigma
            Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T
            
            ## Adam step
            grad = self._adam_update(Gobj, iter, beta_1, beta_2)
            W -= lr * grad

            dif = self.Id - W 
            rhs = self.cov @ dif
            sigma = np.sqrt(np.trace(dif.T @ rhs) / (self.d))
            
            ## Check obj convergence
            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new, _, _ = self._func(W, sigma, mu, s)
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    # pbar.update(max_iter-iter+1)
                    break
                obj_prev = obj_new
            # pbar.update(1)
        return W, sigma, True
    
    def fit(self, X, lambda1, T=5,
            mu_init=1.0, mu_factor=0.1, s=[1.0, .9, .8, .7, .6], 
            warm_iter=3e4, max_iter=6e4, lr=0.0003, 
            checkpoint=1000, beta_1=0.99, beta_2=0.999,
            disable_tqdm=True
        ):
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)
        self.X -= X.mean(axis=0, keepdims=True)
            
        self.cov = X.T @ X / float(self.n)    
        self.W_est = np.zeros((self.d,self.d)).astype(self.dtype) # init W0 at zero matrix
        self.sig_est = np.min(np.linalg.norm(self.X, axis=0) / np.sqrt(self.n)).astype(self.dtype)
        mu = mu_init
        if type(s) == list:
            if len(s) < T: 
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.")    
        
        with tqdm(total=(T-1)*warm_iter+max_iter, disable=disable_tqdm) as pbar:
            for i in range(int(T)):
                lr_adam, success = lr, False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
                while success is False:
                    W_temp, sig_temp, success = self.minimize(self.W_est.copy(), self.sig_est.copy(), mu, inner_iters, s[i], lr=lr_adam, beta_1=beta_1, beta_2=beta_2, pbar=pbar)
                    if success is False:
                        lr_adam *= 0.5
                        s[i] += 0.1
                self.W_est = W_temp
                self.sig_est = sig_temp
                mu *= mu_factor

        return self.W_est, self.sig_est 
    
#===================================#
#   Non-equal variance  CoLiDE-NV   #
#===================================#

class colide_nv:
    
    def __init__(self, dtype=np.float64, seed=0):
        super().__init__()
        np.random.seed(seed)
        self.dtype = dtype
            
    def _score(self, W, sigma):
        dif = self.Id - W 
        rhs = self.cov @ dif
        inv_SigMa = np.diag(1.0/(sigma))
        loss = (np.trace(inv_SigMa @ (dif.T @ rhs)) + np.sum(sigma)) / (2.0)
        G_loss = (-rhs @ inv_SigMa)
        return loss, G_loss

    def _h(self, W, s=1.0):
        M = s * self.Id - W * W
        h = - la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T 
        return h, G_h

    def _func(self, W, sigma, mu, s=1.0):
        score, _ = self._score(W, sigma)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h 
        return obj, score, h
    
    def _adam_update(self, grad, iter, beta_1, beta_2):
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - beta_1 ** iter)
        v_hat = self.opt_v / (1 - beta_2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad
    
    def minimize(self, W, sigma, mu, max_iter, s, lr, tol=1e-6, beta_1=0.99, beta_2=0.999, pbar=None):
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        
        for iter in range(1, max_iter+1):
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < -1e-6):
                if iter == 1 or s <= 0.9:
                    return W, sigma, False
                else:
                    W += lr * grad
                    lr *= .5
                    if lr <= 1e-16:
                        return W, sigma, True
                    W -= lr * grad
                    dif = self.Id - W
                    rhs = self.cov @ dif
                    sigma = np.sqrt(np.diag(dif.T @ rhs))
                    M = sla.inv(s * self.Id - W * W) + 1e-16
            
            inv_SigMa = np.diag(1.0/(sigma))
            G_score = -mu * (self.cov @ (self.Id - W) @ inv_SigMa)
            Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T
            
            ## Adam step
            grad = self._adam_update(Gobj, iter, beta_1, beta_2)
            W -= lr * grad

            dif = self.Id - W
            rhs = self.cov @ dif
            sigma = np.sqrt(np.diag(dif.T @ rhs))
            
            ## Check obj convergence
            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new, _, _ = self._func(W, sigma, mu, s)
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    # pbar.update(max_iter-iter+1)
                    break
                obj_prev = obj_new
            # pbar.update(1)
        return W, sigma, True
    
    def fit(self, X, lambda1, T=5,
            mu_init=1.0, mu_factor=0.1, s=[1.0, .9, .8, .7, .6], 
            warm_iter=3e4, max_iter=6e4, lr=0.0003, 
            checkpoint=1000, beta_1=0.99, beta_2=0.999, w_init=None,
            disable_tqdm=True
        ):
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)
        self.X -= X.mean(axis=0, keepdims=True)
            
        self.cov = X.T @ X / float(self.n)
        if w_init is None:    
            self.W_est = np.zeros((self.d,self.d)).astype(self.dtype) # init W0 at zero matrix
            self.sig_est = (np.linalg.norm(self.X, axis=0) / np.sqrt(self.n)).astype(self.dtype)
        else:
            self.W_est = np.copy(w_init).astype(self.dtype)
            self.sig_est = (np.linalg.norm(self.X @ (self.Id - w_init), axis=0) / np.sqrt(self.n)).astype(self.dtype)

        mu = mu_init
        if type(s) == list:
            if len(s) < T: 
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.")    
        
        with tqdm(total=(T-1)*warm_iter+max_iter, disable=disable_tqdm) as pbar:
            for i in range(int(T)):
                lr_adam, success = lr, False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
                while success is False:
                    W_temp, sig_temp, success = self.minimize(self.W_est.copy(), self.sig_est.copy(), mu, inner_iters, s[i], lr=lr_adam, beta_1=beta_1, beta_2=beta_2, pbar=pbar)
                    if success is False:
                        lr_adam *= 0.5
                        s[i] += 0.1
                self.W_est = W_temp
                self.sig_est = sig_temp
                mu *= mu_factor
                
        return self.W_est, self.sig_est
    

class DAGMA_linear:
    
    def __init__(self, loss_type, verbose=False, dtype=np.float64, seed=0):
        super().__init__()
        np.random.seed(seed)
        losses = ['l2', 'logistic']
        assert loss_type in losses, f"loss_type should be one of {losses}"
        self.loss_type = loss_type
        self.dtype = dtype
        self.vprint = print if verbose else lambda *a, **k: None
            
    def _score(self, W):
        """Evaluate value and gradient of the score function."""
        if self.loss_type == 'l2':
            dif = self.Id - W 
            rhs = self.cov @ dif
            loss = 0.5 * np.trace(dif.T @ rhs)
            G_loss = -rhs
        elif self.loss_type == 'logistic':
            R = self.X @ W
            loss = 1.0 / self.n * (np.logaddexp(0, R) - self.X * R).sum()
            G_loss = (1.0 / self.n * self.X.T) @ sigmoid(R) - self.cov
        return loss, G_loss

    def _h(self, W, s=1.0):
        """Evaluate value and gradient of the logdet acyclicity constraint."""
        M = s * self.Id - W * W
        h = - la.slogdet(M)[1] + self.d * np.log(s)
        G_h = 2 * W * sla.inv(M).T 
        return h, G_h
    
    def dagness(self, W, s=1):
        value, _ = self._h(W, s)
        return value

    def _func(self, W, mu, s=1.0):
        """Evaluate value of the penalized objective function."""
        score, _ = self._score(W)
        h, _ = self._h(W, s)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + h 
        return obj, score, h
    
    def _adam_update(self, grad, iter, beta_1, beta_2):
        self.opt_m = self.opt_m * beta_1 + (1 - beta_1) * grad
        self.opt_v = self.opt_v * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m / (1 - beta_1 ** iter)
        v_hat = self.opt_v / (1 - beta_2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad
    
    def minimize(self, W, mu, max_iter, s, lr, tol=1e-6, beta_1=0.99, beta_2=0.999, pbar=None):
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        self.vprint(f'\n\nMinimize with -- mu:{mu} -- lr: {lr} -- s: {s} -- l1: {self.lambda1} for {max_iter} max iterations')
        
        for iter in range(1, max_iter+1):
            ## Compute the (sub)gradient of the objective
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < 0): # sI - W o W is not an M-matrix
                if iter == 1 or s <= 0.9:
                    self.vprint(f'W went out of domain for s={s} at iteration {iter}')
                    return W, False
                else:
                    W += lr * grad
                    lr *= .5
                    if lr <= 1e-16:
                        return W, True
                    W -= lr * grad
                    M = sla.inv(s * self.Id - W * W) + 1e-16
                    self.vprint(f'Learning rate decreased to lr: {lr}')
            
            if self.loss_type == 'l2':
                G_score = -mu * self.cov @ (self.Id - W) 
            elif self.loss_type == 'logistic':
                G_score = mu / self.n * self.X.T @ sigmoid(self.X @ W) - mu * self.cov
            Gobj = G_score + mu * self.lambda1 * np.sign(W) + 2 * W * M.T
            
            ## Adam step
            grad = self._adam_update(Gobj, iter, beta_1, beta_2)
            W -= lr * grad
            
            ## Check obj convergence
            if iter % self.checkpoint == 0 or iter == max_iter:
                obj_new, score, h = self._func(W, mu, s)
                self.vprint(f'\nInner iteration {iter}')
                self.vprint(f'\th(W_est): {h:.4e}')
                self.vprint(f'\tscore(W_est): {score:.4e}')
                self.vprint(f'\tobj(W_est): {obj_new:.4e}')
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    # pbar.update(max_iter-iter+1)
                    break
                obj_prev = obj_new
            # pbar.update(1)
        return W, True
    
    def fit(self, X, lambda1, w_threshold=0.3, T=5,
            mu_init=1.0, mu_factor=0.1, s=[1.0, .9, .8, .7, .6], 
            warm_iter=3e4, max_iter=6e4, lr=0.0003, 
            checkpoint=1000, beta_1=0.99, beta_2=0.999,
            disable_tqdm=True
        ):
        ## INITALIZING VARIABLES 
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)
        
        if self.loss_type == 'l2':
            self.X -= X.mean(axis=0, keepdims=True)
            
        self.cov = X.T @ X / float(self.n)    
        self.W_est = np.zeros((self.d,self.d)).astype(self.dtype) # init W0 at zero matrix
        mu = mu_init
        if type(s) == list:
            if len(s) < T: 
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.")    
        
        ## START DAGMA
        with tqdm(total=(T-1)*warm_iter+max_iter, disable=disable_tqdm) as pbar:
            for i in range(int(T)):
                self.vprint(f'\nIteration -- {i+1}:')
                lr_adam, success = lr, False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
                while success is False:
                    W_temp, success = self.minimize(self.W_est.copy(), mu, inner_iters, s[i], lr=lr_adam, beta_1=beta_1, beta_2=beta_2, pbar=pbar)
                    if success is False:
                        self.vprint(f'Retrying with larger s')
                        lr_adam *= 0.5
                        s[i] += 0.1
                self.W_est = W_temp
                mu *= mu_factor
        
        ## Store final h and score values and threshold
        self.h_final, _ = self._h(self.W_est)
        self.score_final, _ = self._score(self.W_est)
        # self.W_est[np.abs(self.W_est) < w_threshold] = 0
        return self.W_est


if __name__ == '__main__':
    import utils
    from timeit import default_timer as timer
    # dagu.set_random_seed(1)
    
    n, d, s0 = 500, 20, 20 # the ground truth is a DAG of 20 nodes and 20 edges in expectation
    graph_type, sem_type = 'ER', 'gauss'
    
    B_true = dagu.simulate_dag(d, s0, graph_type)
    W_true = dagu.simulate_parameter(B_true)
    X = dagu.simulate_linear_sem(W_true, n, sem_type)
    
    model = DAGMA_linear(loss_type='l2')
    start = timer()
    W_est = model.fit(X, lambda1=0.02)
    end = timer()
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)
    print(f'time: {end-start:.4f}s')
    
    # Store outputs and ground-truth
    np.savetxt('W_true.csv', W_true, delimiter=',')
    np.savetxt('W_est.csv', W_est, delimiter=',')
    np.savetxt('X.csv', X, delimiter=',')



class Nonneg_dagma():
    """
    Projected Gradient Descet algorithm for learning DAGs with DAGMA acyclicity constraint
    """
    def __init__(self, primal_opt='pgd', acyclicity='logdet'):
        self.acyc_const = acyclicity
        if acyclicity == 'logdet':
            self.dagness = self.logdet_acyc_
            self.gradient_acyclic = self.logdet_acyclic_grad_
        elif acyclicity == 'matexp':
            self.dagness = self.matexp_acyc_
            self.gradient_acyclic = self.matexp_acyclic_grad_
        else:
            raise ValueError('Unknown acyclicity constraint')

        self.opt_type = primal_opt
        if primal_opt in ['pgd', 'adam']:
            self.minimize_primal = self.proj_grad_desc_
        elif primal_opt == 'fista':
            self.minimize_primal = self.acc_proj_grad_desc_
        else:
            raise ValueError('Unknown solver type for primal problem')

    def logdet_acyc_(self, W):
        """
        Evaluates the acyclicity constraint
        """
        return self.N * np.log(self.s) - la.slogdet(self.s*self.Id - W)[1]
    
    def logdet_acyclic_grad_(self, W):
        return la.inv(self.s*self.Id - W).T
    
    def matexp_acyc_(self, W):
        # Clip W to prevent overflowing
        entry_limit = np.maximum(10, 5e2/W.shape[0])
        W = np.clip(W, -entry_limit, entry_limit)
        return np.trace(sla(W)) - self.N

    def matexp_acyclic_grad_(self, W):
        # Clippling gradient to prevent overflow
        return np.clip(sla(W).T, -1e7, 1e7)



    def fit(self, X, alpha, lamb, stepsize, s=1, max_iters=1000, checkpoint=250, tol=1e-6,
            beta1=.99, beta2=.999, Sigma=1, track_seq=False, verb=False):
        
        self.init_variables_(X, track_seq, s, Sigma, beta1, beta2, verb)
        self.W_est, _ = self.minimize_primal(self.W_est, lamb, alpha, stepsize, max_iters,
                                             checkpoint, tol, track_seq)
        
        return self.W_est

    def init_variables_(self, X, track_seq, s, Sigma, beta1, beta2, verb):
        self.M, self.N = X.shape
        self.Cx = X.T @ X / self.M
        self.W_est = np.zeros_like(self.Cx)
        self.verb = verb

        if np.isscalar(Sigma):
            self.Sigma_inv = 1 / Sigma * np.ones((self.N))
        elif Sigma.ndim == 1:
            self.Sigma_inv = 1 / Sigma
        elif Sigma.ndim == 2:
            assert np.all(Sigma == np.diag(np.diag(Sigma))), 'Sigma must be a diagonal matrix'
            self.Sigma_inv = 1 / np.diag(Sigma)
        else:
            raise ValueError("Sigma must be a scalar, vector or diagonal Matrix")

        self.Id = np.eye(self.N)
        self.s = s

        # For Adam
        self.opt_m, self.opt_v = 0, 0
        self.beta1, self.beta2 = beta1, beta2
        
        self.acyclicity = []
        self.diff = []
        self.seq_W = [] if track_seq else None

    def compute_gradient_(self, W, lamb, alpha):        
        G_loss = self.Cx @(W - self.Id) * self.Sigma_inv / 2 + lamb
        G_acyc = self.gradient_acyclic(W)
        return G_loss + alpha*G_acyc
    
    def compute_adam_grad_(self, grad, iter):
        self.opt_m = self.opt_m * self.beta1 + (1 - self.beta1) * grad
        self.opt_v = self.opt_v * self.beta2 + (1 - self.beta2) * (grad ** 2)
        m_hat = self.opt_m / (1 - self.beta1 ** iter)
        v_hat = self.opt_v / (1 - self.beta2 ** iter)
        grad = m_hat / (np.sqrt(v_hat) + 1e-8)
        return grad


    def proj_grad_step_(self, W, alpha, lamb, stepsize, iter):
        G_obj_func = self.compute_gradient_(W, lamb, alpha)
        if self.opt_type == 'adam':
            G_obj_func = self.compute_adam_grad_(G_obj_func, iter+1)
        W_est = np.maximum(W - stepsize*G_obj_func, 0)

        # Ensure non-negative acyclicity
        if self.acyc_const == 'logdet':
            acyc = self.dagness(W_est)        
            if acyc < -1e-12:
                eigenvalues, _ = np.linalg.eig(W_est)
                max_eigenvalue = np.max(np.abs(eigenvalues))
                W_est = W_est/(max_eigenvalue + 1e-2)
                acyc = self.dagness(W_est)

                stepsize /= 2
                if self.verb:
                    print('Negative acyclicity. Projecting and reducing stepsize to: ', stepsize)

                assert acyc > -1e-12, f'Acyclicity is negative: {acyc}'
        
        return W_est, stepsize

    def tack_variables_(self, W, W_prev, track_seq):
        norm_W_prev = la.norm(W_prev)
        norm_W_prev = norm_W_prev if norm_W_prev != 0 else 1
        self.diff.append(la.norm(W - W_prev) / norm_W_prev)
        if track_seq:
            self.seq_W.append(W)
            self.acyclicity.append(self.dagness(W))


    def proj_grad_desc_(self, W, lamb, alpha, stepsize, max_iters, checkpoint, tol,
                        track_seq):
        W_prev = W.copy()
        for i in range(max_iters):
            W, stepsize = self.proj_grad_step_(W_prev, alpha, lamb, stepsize, i)

            # Update tracking variables
            self.tack_variables_(W, W_prev, track_seq)

            # Check convergence
            if i % checkpoint == 0 and self.diff[-1] <= tol:
                break
    
            W_prev = W.copy()
        
        return W, stepsize

    def acc_proj_grad_desc_(self, W, lamb, alpha, stepsize, max_iters, checkpoint, tol,
                            track_seq):
        W_prev = W.copy()
        W_fista = np.copy(W) 
        t_k = 1
        for i in range(max_iters):
            W, stepsize = self.proj_grad_step_(W_fista, alpha, lamb, stepsize, i)
            t_next = (1 + np.sqrt(1 + 4*t_k**2))/2
            W_fista = W + (t_k - 1)/t_next*(W - W_prev)

            # Update tracking variables
            self.tack_variables_(W, W_prev, track_seq)

            # Check convergence
            if i % checkpoint == 0 and self.diff[-1] <= tol:
                break

            W_prev = W
            t_k = t_next
        
        return W, stepsize



class MetMulDagma(Nonneg_dagma):
    """
    Method of ultipliers algorithm for learning DAGs with DAGMA acyclicity constraint
    """
    def fit(self, X, lamb, stepsize, s=1, iters_in=1000, iters_out=10, checkpoint=250, tol=1e-6,
            beta=5, gamma=.25, rho_0=1, alpha_0=.1, track_seq=False, dec_step=None,
            beta1=.99, beta2=.999, Sigma=1, verb=False):

        self.init_variables_(X, rho_0, alpha_0, track_seq, s, Sigma, beta1, beta2,  verb)        
        dagness_prev = self.dagness(self.W_est)

        for i in range(iters_out):
            # Estimate W
            self.W_est, stepsize = self.minimize_primal(self.W_est, lamb, self.alpha, stepsize, iters_in,
                                                 checkpoint, tol, track_seq)

            # Update augmented Lagrangian parameters
            dagness = self.dagness(self.W_est)
            self.rho = beta*self.rho if dagness > gamma*dagness_prev else self.rho
            
            # Update Lagrange multiplier
            self.alpha += self.rho*dagness

            dagness_prev = dagness

            if dec_step:
                stepsize *= dec_step

            if verb:
                print(f'- {i+1}/{iters_out}. Diff: {self.diff[-1]:.6f} | Acycl: {dagness:.6f}' +
                      f' | Rho: {self.rho:.3f} - Alpha: {self.alpha:.3f} - Step: {stepsize:.4f}')
                                    
        return self.W_est  

    def compute_gradient_(self, W, lamb, alpha):
        G_loss = self.Cx @(W - self.Id) * self.Sigma_inv / 2 + lamb
        acyc_val = self.dagness(W)
        G_acyc = self.gradient_acyclic(W)
        return G_loss + (alpha + self.rho*acyc_val)*G_acyc

    def init_variables_(self, X, rho_init, alpha_init, track_seq, s, Sigma, beta1, beta2,  verb):
        super().init_variables_(X, track_seq, s, Sigma, beta1, beta2,  verb)
        self.rho = rho_init
        self.alpha = alpha_init


import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid


def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


