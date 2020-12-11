import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma
import math


class VI:
    def __init__(self, data, actual_mu, actual_sigma):
        self.actual_mu = actual_mu
        self.actual_variance = actual_sigma
        self.mu_0 = 0
        self.a_0 = 0
        self.b_0 = 0
        self.lambda_0 = 0
        self.expected_mu = 0
        self.expected_mu_2nd = 0
        self.expected_tau = 0
        self.mu = 0
        self.alpha = 0
        self.n = 0
        self.data = data
        N = len(data)
        sample_mean = np.mean(data)
        var = np.sum((data - sample_mean) ** 2)
        self.actual_a = 0.5*N
        self.actual_b = 0.5*var
        self.actual_tau = self.actual_a/self.actual_b
        self.actual_lambda = N

    def initialize(self, lambda_n):
        self.n = len(self.data)
        self.expected_mu = np.mean(self.data)
        print("exp mu", self.expected_mu)
        self.mu = np.mean(self.data)
        print("mu", self.mu)
        self.expected_mu_2nd = 1 / lambda_n + self.expected_mu ** 2
        print("mu 2nd moment", self.expected_mu_2nd)
        self.alpha = self.a_0 + self.n / 2
        print("alpha", self.alpha)

    def update_tau_elbo(self, lambda_n):
        print("lambda_n from beta update", lambda_n)
        print("n from tau update", self.n)
        print("exp tau from tau update", self.expected_tau)
        self.expected_mu_2nd = self.mu**2 + 1/lambda_n
        #sum = 0
        beta = 1 / 2 * sum([x * x + 1 / (lambda_n * self.n) + self.mu ** 2 - 2 * self.mu * x for x in self.data])
        #for x in self.data:
         #   sum += x ** 2 + self.expected_mu_2nd - 2 * self.expected_mu * x
        #beta = self.b_0 + 0.5 * self.lambda_0 * (
         #           self.expected_mu_2nd - self.mu_0 ** 2 - 2 * self.expected_mu * self.mu_0) + 0.5 * sum
        #beta = 0.5*sum
        #self.expected_tau = self.alpha / beta
        #self.lambda_n = (self.lambda_0+self.n)*self.expected_tau
        #print("lamnda_n", self.lambda_n)
        return beta

    def update_mu_elbo(self, beta):
        print("expected_mu_2nd", self.expected_mu_2nd)
        #tau = (self.lambda_0 + self.n) * self.alpha / beta
        self.expected_tau = self.alpha / beta
        lambda_n = (self.lambda_0 + self.n) * self.expected_tau
        return lambda_n

    def fit(self, lambda_n):
        self.initialize(lambda_n)
        #old_exp_tau = tau
        diff_tau = 1000
        diff_beta = 1000
        old_beta = 0
        #print(self.actual_mu, self.actual_variance)
        while diff_tau and diff_beta > 10e-3:
            new_beta = self.update_tau_elbo(lambda_n)
            new_lambda_n = self.update_mu_elbo(new_beta)
            print("beta", new_beta)
            #print("1/tau", 1 / new_exp_tau)
            print("1/expected tau", 1 / self.expected_tau)
            plot_posterior(self.actual_mu, self.actual_tau, self.actual_lambda, self.actual_a, self.actual_b)
            plt.subplot()
            plot_posterior(self.mu, self.expected_tau, lambda_n, self.alpha, new_beta, color='green')
            plt.show()
            diff_lambda = abs(new_lambda_n-lambda_n)
            diff_beta = abs(new_beta - old_beta)
            print("diff_tau", diff_lambda)
            print("diff_beta", diff_beta)
            old_beta = new_beta
            lambda_n = new_lambda_n
        print("converged")
        print("data to plot functions: mu %f tau %f lambda %f a %f b %f" % (
        self.actual_mu, self.actual_tau, self.actual_lambda, self.actual_a, self.actual_b))
        print("estimated data to plot functions: mu %f tau %f lambda %f a %f b %f" % (
        self.mu, self.expected_tau, lambda_n, self.alpha, new_beta))
        #print("converged after %d iterations" % iter)
        #plot_posterior(self.actual_mu, self.actual_tau, self.actual_lambda, self.actual_a, self.actual_b)
        #plt.subplot()
        #plot_posterior(self.mu, self.expected_tau, lambda_n, self.alpha, new_beta, color='green')
        #plt.show()
        return self.mu, new_lambda_n

    #def plot_inferred_posterior(self):



def compute_normal_prob(x, mean, precision):
    dist = norm.pdf(x, mean, math.sqrt(1/precision))
    return dist

def compute_gamma_prob(tau, a, b):
    dist = gamma.pdf(tau, a, loc=0, scale=(1/b))
    return dist


def plot_posterior(mu, tau, lambda_, a_n, b_n, color='blue'):
    mu_range = np.linspace(1, 3, 100)
    tau_range = np.linspace(0, 0.1, 100)
    mu_grid, tau_grid = np.meshgrid(mu_range, tau_range, indexing='ij')
    z = np.zeros_like(mu_grid)

    for i, mu_ in enumerate(mu_range):
        for j, tau_ in enumerate(tau_range):
            z[i, j] = compute_normal_prob(mu_, mu, lambda_*tau)*compute_gamma_prob(tau_, a_n, b_n)
    plt.contour(mu_grid, tau_grid, z, colors=color)
    plt.title("Posterior distribution of inferred parameters, blue: true posterior, green: inferred posterior")
    #plt.legend()
    plt.xlabel("Mean")
    plt.ylabel("Precision")






mu = 2
sigma = 4
sample = 110
np.random.seed(50)
x = np.random.normal(loc=mu, scale=sigma, size=sample)

vi = VI(x, mu, sigma)
start_lambda = np.random.uniform(0, 1)
#start_tau = 0.4
print('start tau', start_lambda)
vi.fit(start_lambda)

