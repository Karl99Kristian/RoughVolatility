# RoughVolatility
Implementation of generalized Bergomi model with regime switching market price of volatility risk


This code models the forward variance process given by
$$V_u = \mathbb{E}\left[V_u\mid\mathcal{F}_t\right]\mathcal{E}\left(\eta(K*\eta dW)(u)\right),$$
under a regime swiching market price of volatility risk given by
$$\lambda_s=\theta(\mu_s-X_s)$$
where $X$ is a affine volterra type Ornstein-Uhlenbeck process with mean reversion towards $\mu$ goverend by a Markov Chain. For an introduction see [1] or the pdf in `/material`

For now only the fractional kernel is implemented.

There are main files for estimation in the direct simulation, study of moment error with an approximate simulation of the VIX and tests.

## Setup and requirements
Doing the following should make the code run smoothly.
`$PATH$` is the directory that RoughVolatility is in. 

```
python -m venv env

echo "export PYTHONPATH=$PATH$" >> .env

printf "\n# Adding this command to read local .env file" >> env/bin/activate
printf "\nexport \$(grep -v '^#' .env | xargs)" >> env/bin/activate

. env/bin/activate

pip install -r requirements.txt
```
Manually imported in the code is the Mittag-Leffler function by K. Hinsen[2].



## References
[1]: Guerreiro, H. and Guerra, J. (2022). ”VIX pricing in the rBergomi model under a regime switching change of measure”, [https://arxiv.org/pdf/2201.10391.pdf](https://arxiv.org/pdf/2201.10391.pdf).

[2]: Hinsen, K. (2017). ”The Mittag-Leffler function in Python”, [https://github.com/
khinsen/mittag-leffler](https://github.com/khinsen/mittag-leffler).

