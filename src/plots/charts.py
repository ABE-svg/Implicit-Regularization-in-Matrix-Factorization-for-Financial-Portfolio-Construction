import matplotlib.pyplot as plt
from ..algorithms.optim_utils import eigenvalues_desc, singular_values_desc

def plot_trace(histories):
    for name, h in histories.items():
        plt.plot(h.steps, h.trace, label=name)
    plt.legend(); plt.title("trace(X)"); plt.show()

def plot_spectrum_X(Xs):
    for name, X in Xs.items():
        plt.semilogy(eigenvalues_desc(X), label=name)
    plt.legend(); plt.title("Eigen spectrum of X"); plt.show()

def plot_spectrum_U(Us):
    for name, U in Us.items():
        plt.semilogy(singular_values_desc(U), label=name)
    plt.legend(); plt.title("Singular values of U"); plt.show()
