
# tvhmm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

# ---- Utils ----
def diag_gaussian_logprob(y, mean, log_var):
    # y: [obs_dim], mean/log_var: [K, obs_dim] or [obs_dim]
    # returns log p(y|k) for each k: [K]
    # stable diagonal Gaussian log-density
    diff = y - mean                     # [K, D]
    inv_var = torch.exp(-log_var)       # [K, D]
    quad = (diff * diff * inv_var).sum(-1)    # [K]
    logdet = log_var.sum(-1)                  # [K]
    D = y.shape[-1]
    return -0.5 * (D * torch.log(torch.tensor(2.0 * torch.pi)) + logdet + quad)

def logsumexp_pairwise(alpha_prev, log_trans):  # shapes: [K], [K,K] (i->j)
    # returns vector over j: logsum_i alpha_prev[i] + log_trans[i,j]
    # -> [K]
    return torch.logsumexp(alpha_prev.unsqueeze(1) + log_trans, dim=0)

# ---- Model ----
class TimeVaryingHMM(nn.Module):
    """
    K-state HMM with time-varying transition probabilities:
      P(z_t = j | z_{t-1}=i, x_t) = softmax_j( b[i,j] + x_t^T W[i,j,:] )
    Emissions: y_t | z_t = k ~ N( mu_k, diag(sigma_k^2) )

    Args:
        K: number of hidden states
        D_x: covariate dimension
        D_y: observation dimension
    """
    def __init__(self, K: int, D_x: int, D_y: int):
        super().__init__()
        self.K, self.D_x, self.D_y = K, D_x, D_y

        # Initial state logits (softmax -> pi0)
        self.init_logits = nn.Parameter(torch.zeros(K))

        # Transition parameters: W[i,j,:] and b[i,j]
        self.W = nn.Parameter(torch.zeros(K, K, D_x))
        self.b = nn.Parameter(torch.zeros(K, K))

        # Emission parameters: state-wise means and log-variances
        self.mu = nn.Parameter(torch.randn(K, D_y) * 0.1)
        self.raw_log_var = nn.Parameter(torch.zeros(K, D_y))  # unconstrained; we’ll clamp for stability

    # --- Core probabilities ---
    def log_init(self):
        return F.log_softmax(self.init_logits, dim=0)  # [K]

    def log_trans(self, x_t):
        """
        x_t: [D_x]
        returns log transition matrix log P(j | i, x_t): [K, K] with rows summing to 1 in probability space
        """
        # logits[i,j] = b[i,j] + x_t^T W[i,j,:]
        logits = self.b + torch.tensordot(self.W, x_t, dims=([2],[0]))  # [K,K]
        return F.log_softmax(logits, dim=1)  # softmax across 'to' state j

    def log_emission(self, y_t):
        """
        y_t: [D_y] -> log p(y_t | z_t=k) for each k: [K]
        """
        log_var = torch.clamp(self.raw_log_var, min=-6.0, max=6.0)  # keep vars in a reasonable range
        return diag_gaussian_logprob(y_t, self.mu, log_var)

    # --- Forward (log-domain) ---
    def sequence_loglik(self, X, Y):
        """
        X: [T, D_x] covariates (can include intercept if you want)
        Y: [T, D_y] observations
        returns scalar log-likelihood log p(Y | X, params)
        """
        T = Y.shape[0]
        log_pi0 = self.log_init()                  # [K]
        log_em0 = self.log_emission(Y[0])          # [K]
        alpha = log_pi0 + log_em0                  # [K]

        for t in range(1, T):
            log_A = self.log_trans(X[t])           # [K,K]
            log_em = self.log_emission(Y[t])       # [K]
            alpha = log_em + logsumexp_pairwise(alpha, log_A)  # [K]

        return torch.logsumexp(alpha, dim=0)       # scalar

    # --- Viterbi decoding ---
    @torch.no_grad()
    def viterbi(self, X, Y):
        T = Y.shape[0]
        log_pi0 = self.log_init()
        log_em0 = self.log_emission(Y[0])
        delta = log_pi0 + log_em0        # [K]
        psi = torch.zeros(T, self.K, dtype=torch.long) - 1

        for t in range(1, T):
            log_A = self.log_trans(X[t])  # [K,K]
            log_em = self.log_emission(Y[t])  # [K]
            scores = delta.unsqueeze(1) + log_A  # [K,K]
            best_prev = scores.argmax(dim=0)     # [K]
            delta = log_em + scores.max(dim=0).values
            psi[t] = best_prev

        # backtrack
        zT = delta.argmax().item()
        path = [zT]
        for t in range(T-1, 0, -1):
            zT = psi[t, zT].item()
            path.append(zT)
        path.reverse()
        return path  # list of length T, elements in {0,...,K-1}

# ---- Tiny example: synthesize data and train ----
if __name__ == "__main__":
    torch.manual_seed(0)

    K, D_x, D_y, T = 3, 2, 1, 300

    # Build a ground-truth model to sample from
    true_model = TimeVaryingHMM(K, D_x, D_y)
    with torch.no_grad():
        true_model.init_logits.copy_(torch.tensor([1.0, 0.0, -1.0]))
        true_model.b.copy_(torch.tensor([[2.0, -1.0, -1.0],
                                         [-1.0, 2.0, -1.0],
                                         [-1.0, -1.0, 2.0]]))
        true_model.W.copy_(torch.tensor([
            [[ 1.0, -0.5], [ 0.5, 0.0],  [-0.5, 0.5]],
            [[-0.5,  0.5], [ 1.0, -0.5], [ 0.5, 0.0]],
            [[ 0.0,  0.5], [-0.5, 0.5],  [ 1.0, -0.5]],
        ]))
        true_model.mu.copy_(torch.tensor([[-2.0],[0.0],[2.0]]))  # well-separated means
        true_model.raw_log_var.fill_(-1.0)  # exp(-1) ~ 0.37 variance

    # Generate covariates X and a sequence (z, y)
    X = torch.randn(T, D_x)
    X[:, 0] = 1.0  # intercept term
    z = torch.zeros(T, dtype=torch.long)
    y = torch.zeros(T, D_y)

    @torch.no_grad()
    def sample_trans_row(logA_row):
        # logA_row: [K] for a fixed from-state i -> categorical over j
        return torch.distributions.Categorical(logits=logA_row).sample()

    with torch.no_grad():
        pi0 = F.softmax(true_model.init_logits, dim=0)
        z[0] = torch.distributions.Categorical(pi0).sample()
        # emission
        for t in range(T):
            # sample y_t
            mean = true_model.mu[z[t]]
            log_var = torch.clamp(true_model.raw_log_var[z[t]], -6, 6)
            cov = torch.diag_embed(torch.exp(log_var))
            y[t] = MultivariateNormal(mean, covariance_matrix=cov).sample()
            # sample next z
            if t < T-1:
                logA = true_model.log_trans(X[t+1])   # use x_{t+1} for transition into z_{t+1}
                z[t+1] = sample_trans_row(logA[z[t]])

    # Fit a fresh model
    model = TimeVaryingHMM(K, D_x, D_y)
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)

    X_train, Y_train = X, y
    for step in range(800):
        opt.zero_grad()
        nll = -model.sequence_loglik(X_train, Y_train)  # negative log-likelihood
        # mild L2 on W to reduce overfitting drift
        reg = 1e-3 * (model.W ** 2).mean()
        loss = nll + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (step+1) % 100 == 0:
            print(f"step {step+1:4d} | NLL: {nll.item():.3f}")

    # Decode most likely state sequence
    path = model.viterbi(X_train, Y_train)
    print("First 20 decoded states:", path[:20])
