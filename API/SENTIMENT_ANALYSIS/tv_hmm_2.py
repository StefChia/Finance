
#FOLLOWING MATH
# tvhmm_derivation.py
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Model:
- Hidden states: z_t ∈ {1..K}
- Covariates:    x_t ∈ ℝ^{D_x}
- Observations:  y_t ∈ ℝ^{D_y}

Initial:
  π0(k) = softmax_k(η_k)

Transitions (time-varying, logit-softmax):
  logits_{i→j}(t) = b_{ij} + x_t^T W_{ij}
  A_t(i,j) = softmax_j logits_{i→j}(t)

Emissions (diag-Gaussian):
  y_t | z_t = k ~ N( μ_k, diag(σ_k^2) )

Forward recursion (log-domain):
  α_1(j) = log π0(j) + log p(y_1 | z_1=j)
  α_t(j) = log p(y_t | z_t=j) + logsumexp_i [ α_{t-1}(i) + log A_t(i,j) ]
  log p(y_{1:T}) = logsumexp_j α_T(j)
"""

# ---------- small numerics helpers ----------
LOG2PI = torch.log(torch.tensor(2.0 * torch.pi))

def diag_gauss_logpdf(y, mean, log_var):
    # y: [D], mean/log_var: [K,D]
    diff = y - mean                          # [K,D]
    inv_var = torch.exp(-log_var)            # [K,D]
    quad = (diff * diff * inv_var).sum(-1)   # [K]
    logdet = log_var.sum(-1)                 # [K]
    D = y.shape[-1]
    return -0.5 * (D * LOG2PI + logdet + quad)  # [K]

# ---------- model ----------
class TVHMM(nn.Module):
    def __init__(self, K: int, D_x: int, D_y: int):
        super().__init__()
        self.K, self.D_x, self.D_y = K, D_x, D_y

        # Initial logits η_k  (π0 = softmax(η))
        self.init_logits = nn.Parameter(torch.zeros(K))

        # Transition parameters W[i,j,:], b[i,j]
        self.W = nn.Parameter(torch.zeros(K, K, D_x))
        self.b = nn.Parameter(torch.zeros(K, K))

        # Emission parameters μ_k, log σ_k^2
        self.mu = nn.Parameter(torch.randn(K, D_y) * 0.1)
        self.log_var_unconstrained = nn.Parameter(torch.zeros(K, D_y))

    # ---- exact pieces that correspond to formulas ----
    def log_pi0(self):
        """ log π0(k) """
        return F.log_softmax(self.init_logits, dim=0)  # [K]

    def transition_logits(self, x_t):
        """
        logits_{i→j}(t) = b_{ij} + x_t^T W_{ij}
        Returns: [K,K] matrix (i→j) of logits (not normalized).
        """
        # tensordot over feature dim -> [K,K]
        return self.b + torch.tensordot(self.W, x_t, dims=([2], [0]))

    def log_A(self, x_t):
        """
        log A_t(i,j) = log softmax_j logits_{i→j}(t)
        (row-wise softmax over 'to' index j)
        """
        logits = self.transition_logits(x_t)  # [K,K]
        return F.log_softmax(logits, dim=1)   # [K,K]

    def log_emission(self, y_t):
        """
        log p(y_t | z_t = k)  for each k
        """
        log_var = torch.clamp(self.log_var_unconstrained, -6.0, 6.0)  # numeric safety
        return diag_gauss_logpdf(y_t, self.mu, log_var)  # [K]

    # ---- forward (sum-product) in log domain, step-by-step ----
    def forward_messages(self, X, Y):
        """
        Compute α_t(j) for t=1..T and return:
          - alphas: list of length T, each [K]
          - loglik: scalar log p(Y|X)
        """
        T = Y.shape[0]
        K = self.K

        # t = 1:
        log_pi0 = self.log_pi0()          # [K]
        log_em1 = self.log_emission(Y[0]) # [K]
        alpha = log_pi0 + log_em1         # [K]
        alphas = [alpha]

        # t = 2..T:
        for t in range(1, T):
            log_A_t = self.log_A(X[t])      # [K,K] (i→j)
            log_em_t = self.log_emission(Y[t])  # [K]
            # α_t(j) = log_em_t(j) + logsumexp_i [ α_{t-1}(i) + log A_t(i,j) ]
            # compute scores over i for each j
            scores_ij = alpha.unsqueeze(1) + log_A_t   # [K,K]
            alpha = log_em_t + torch.logsumexp(scores_ij, dim=0)  # [K]
            alphas.append(alpha)

        loglik = torch.logsumexp(alphas[-1], dim=0)  # scalar
        return alphas, loglik

    def sequence_loglik(self, X, Y):
        return self.forward_messages(X, Y)[1]

    # ---- Viterbi (max-product) with backpointers ----
    @torch.no_grad()
    def viterbi(self, X, Y):
        T = Y.shape[0]
        delta = self.log_pi0() + self.log_emission(Y[0])  # [K]
        psi = torch.full((T, self.K), -1, dtype=torch.long)

        for t in range(1, T):
            log_A_t = self.log_A(X[t])            # [K,K]
            log_em_t = self.log_emission(Y[t])    # [K]
            scores_ij = delta.unsqueeze(1) + log_A_t  # [K,K]
            best_prev = scores_ij.argmax(dim=0)       # [K]
            delta = log_em_t + scores_ij.max(dim=0).values
            psi[t] = best_prev

        z_T = delta.argmax().item()
        path = [z_T]
        for t in range(T-1, 0, -1):
            z_T = psi[t, z_T].item()
            path.append(z_T)
        path.reverse()
        return path

# ---------- tiny runnable example ----------
if __name__ == "__main__":
    torch.manual_seed(0)
    K, D_x, D_y, T = 3, 2, 1, 200

    # covariates (include intercept as x[:,0]=1)
    X = torch.randn(T, D_x)
    X[:, 0] = 1.0

    # build a ground-truth model to sample from
    true_model = TVHMM(K, D_x, D_y)
    with torch.no_grad():
        true_model.init_logits.copy_(torch.tensor([0.7, 0.0, -0.7]))
        true_model.b.copy_(torch.tensor([[ 1.5, -0.5, -1.0],
                                         [-1.0,  1.5, -0.5],
                                         [-0.5, -1.0,  1.5]]))
        true_model.W.copy_(torch.tensor([
            [[ 0.8, -0.4], [ 0.3,  0.0], [-0.4,  0.4]],
            [[-0.4,  0.4], [ 0.8, -0.4], [ 0.3,  0.0]],
            [[ 0.0,  0.4], [-0.4,  0.4], [ 0.8, -0.4]],
        ]))
        true_model.mu.copy_(torch.tensor([[-2.0],[0.0],[2.0]]))
        true_model.log_var_unconstrained.fill_(-1.0)

    # sample a sequence (z, y)
    z = torch.zeros(T, dtype=torch.long)
    y = torch.zeros(T, D_y)
    with torch.no_grad():
        pi0 = F.softmax(true_model.init_logits, dim=0)
        z[0] = torch.distributions.Categorical(pi0).sample()
        for t in range(T):
            # emission
            log_var = torch.clamp(true_model.log_var_unconstrained[z[t]], -6, 6)
            var = torch.exp(log_var)
            y[t] = torch.randn(D_y) * torch.sqrt(var) + true_model.mu[z[t]]
            # transition
            if t < T-1:
                log_A_next = true_model.log_A(X[t+1])      # i→j
                logits_row = log_A_next[z[t]]               # to-states for current i
                z[t+1] = torch.distributions.Categorical(logits=logits_row).sample()

    # fit a fresh model by maximizing exact log-likelihood
    model = TVHMM(K, D_x, D_y)
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)

    for step in range(600):
        opt.zero_grad()
        nll = -model.sequence_loglik(X, y)
        # mild ridge on W to stabilize
        loss = nll + 1e-3 * (model.W**2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (step+1) % 100 == 0:
            print(f"step {step+1:4d} | NLL: {nll.item():.3f}")

    path = model.viterbi(X, y)
    print("Decoded first 15 states:", path[:15])
