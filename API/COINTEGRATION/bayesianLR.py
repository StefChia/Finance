
import numpy as np
import pymc as pm
import arviz as az

# toy data
rng = np.random.default_rng(0)
n, p = 200, 3
X = rng.normal(size=(n, p))
beta_true = np.array([1.5, -2.0, 0.7])
y = X @ beta_true + rng.normal(scale=1.0, size=n)

with pm.Model() as model:
    beta = pm.Normal("beta", mu=0, sigma=10, shape=p)
    sigma = pm.HalfNormal("sigma", sigma=2.5)
    mu = pm.Deterministic("mu", pm.math.dot(X, beta))
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    idata = pm.sample(1000, tune=1000, target_accept=0.9, random_seed=0)
    ppc = pm.sample_posterior_predictive(idata)

print(az.summary(idata, var_names=["beta","sigma"]))
# Posterior predictive mean:
print(ppc.posterior_predictive["y_obs"].mean(("chain","draw")).values)
