import os
import numpy as np
import matplotlib.pyplot as plt
from ..algorithms.optim_utils import eigenvalues_desc, singular_values_desc

SAVE_DIR = "assets/img"
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_trace(histories, filename="trace.png"):
    plt.figure(figsize=(7,4))
    for name, h in histories.items():
        plt.plot(h.steps, h.trace, label=name)
    plt.xlabel("steps")
    plt.ylabel("trace(X) = ||X||_* (PSD)")
    plt.title("Implicit regularization signal: trace(X)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=200)
    plt.show()
    plt.close()

def plot_spectrum_X(Xs, filename="spectrum_X.png"):
    plt.figure(figsize=(7,4))
    for name, X in Xs.items():
        e = eigenvalues_desc(X)
        plt.semilogy(range(1, len(e)+1), e, marker="o", label=name)
    plt.xlabel("index")
    plt.ylabel("eigenvalue (log)")
    plt.title("Eigenvalue spectrum of covariance X")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=200)
    plt.show()
    plt.close()

def plot_spectrum_U(Us, filename="spectrum_U.png"):
    plt.figure(figsize=(7,4))
    for name, U in Us.items():
        s = singular_values_desc(U)
        plt.semilogy(range(1, len(s)+1), s, marker="o", label=name)
    plt.xlabel("index")
    plt.ylabel("singular value (log)")
    plt.title("Singular values of U")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=200)
    plt.show()
    plt.close()

def plot_cumulative_wealth(wealth_dict, filename="cumulative_wealth.png"):
    plt.figure(figsize=(7,4))
    for name, w in wealth_dict.items():
        plt.plot(w, label=name)
    plt.xlabel("time")
    plt.ylabel("wealth")
    plt.title("Cumulative wealth (test period)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=200)
    plt.show()
    plt.close()

def plot_drawdown(dd_dict, filename="drawdown.png"):
    plt.figure(figsize=(7,4))
    for name, dd in dd_dict.items():
        plt.plot(dd, label=name)
    plt.xlabel("time")
    plt.ylabel("drawdown")
    plt.title("Drawdown (test period)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=200)
    plt.show()
    plt.close()
