import os
import matplotlib.pyplot as plt
from ..algorithms.optim_utils import eigenvalues_desc, singular_values_desc

SAVE_DIR = "assets/img"
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_trace(histories):
    plt.figure()
    for name, h in histories.items():
        plt.plot(h.steps, h.trace, label=name)
    plt.legend()
    plt.title("Trace(X) = Nuclear norm (PSD)")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/trace.png", dpi=200)
    plt.show()
    plt.close()

def plot_spectrum_X(Xs):
    plt.figure()
    for name, X in Xs.items():
        plt.semilogy(eigenvalues_desc(X), label=name)
    plt.legend()
    plt.title("Eigenvalue spectrum of X")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/spectrum_X.png", dpi=200)
    plt.show()
    plt.close()

def plot_spectrum_U(Us):
    plt.figure()
    for name, U in Us.items():
        plt.semilogy(singular_values_desc(U), label=name)
    plt.legend()
    plt.title("Singular values of U")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/spectrum_U.png", dpi=200)
    plt.show()
    plt.close()
