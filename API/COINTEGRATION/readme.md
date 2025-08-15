
Pairs-style PnL is roughly proportional to the negative forward change:
PnL proportional to:  - sign(et)(et+1- et)


**Short answer:** if you’re testing mean-reversion correctly, you should usually see  
`bipolar_ex < bipolar_mid` (both typically ≤ 0) when you compare the **current residual** to the **forward difference**.

**Why:**

For an AR(1):
\[
x_{t+1} = \phi x_t + \varepsilon_{t+1}
\]
implies
\[
\Delta x_{t+1} = x_{t+1} - x_t = (\phi - 1)\,x_t + \varepsilon_{t+1}.
\]

With \(0 < \phi < 1\) (mean-reverting), we have \(\phi - 1 < 0\): large \(x_t\) implies \(\Delta x_{t+1}\) tends to have the **opposite sign**.  
Extremes are more reliably opposite than mid-range ⇒ the **extreme-weighted** concordance is **more negative**.





