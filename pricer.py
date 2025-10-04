import math
import numpy as np
from scipy.stats import norm

def call_europeen_bs(S0, K, T, r, sigma):
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price

def call_europeen_monte_carlo(S0, K, T, r, sigma, n_simulations=100000):
    Z = np.random.standard_normal(n_simulations) # Génération de tirages normaux (Z ~ N(0,1))
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z) # Simulation des prix finaux S_T
    payoff = np.maximum(ST - K, 0)
    call_price = np.exp(-r * T) * np.mean(payoff)
    return call_price

def put_europeen_bs(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return put_price

def put_europeen_monte_carlo(S0, K, T, r, sigma, n_simulations=100000):
    Z = np.random.standard_normal(n_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(K - ST, 0)
    put_price = np.exp(-r * T) * np.mean(payoff)
    return put_price

# --- Paramètres ---
S0, K, T, r, sigma = 100, 90, 1, 0.05, 0.2

# --- Calculs ---
call_bs = call_europeen_bs(S0, K, T, r, sigma)
call_mc = call_europeen_monte_carlo(S0, K, T, r, sigma)
put_bs = put_europeen_bs(S0, K, T, r, sigma)
put_mc = put_europeen_monte_carlo(S0, K, T, r, sigma)

# --- Résultats ---
print(f"\nPrix Black-Scholes : {call_bs:.4f}")
print(f"Prix Monte Carlo   : {call_mc:.4f}")
print(f"Erreur relative    : {abs(call_mc - call_bs) / call_bs * 100:.3f}%\n")

print(f"Prix Black-Scholes : {put_bs:.4f}")
print(f"Prix Monte Carlo   : {put_mc:.4f}")
print(f"Erreur relative    : {abs(put_mc - put_bs) / put_bs * 100:.3f}%")

# --- Parité Call Put ---
membre1_mc = call_mc - put_mc
membre1_bs = call_bs - put_bs
membre2 = S0-K*np.exp(-r*T)

print("\nParité Call Put BS")
print(f"Membre 1 : {membre1_bs:.4f}")
print(f"Membre 2 : {membre2:.4f}")

print("\nParité Call Put MC")
print(f"Membre 1 : {membre1_mc:.4f}")
print(f"Membre 2 : {membre2:.4f}")
