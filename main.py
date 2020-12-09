import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma
import math


class VI():
    def __init__(self, data, actual_mu, actual_sigma):
        self.actual_mu = actual_mu
        self.actual_variance = actual_sigma
        self.mu_0 = 0
        self.a_0 = 0
        self.b_0 = 0
        self.lambda_0 = 0
        ##lambda 10.27 from bishop
        self.lambda_n = 0
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

    def initialize(self, tau):
        self.n = len(self.data)
        self.expected_mu = np.mean(self.data)
        print("exp mu", self.expected_mu)
        self.mu = np.mean(self.data)
        print("mu", self.mu)
        self.expected_mu_2nd = 1 / tau + self.expected_mu ** 2
        print("mu 2nd moment", self.expected_mu_2nd)
        self.alpha = self.a_0 + (self.n + 1) / 2
        print("alpha", self.alpha)

    def update_tau_elbo(self, tau):
        self.expected_mu_2nd = 1 / tau + self.expected_mu ** 2
        sum = 0
        for x in self.data:
            sum += x ** 2 + self.expected_mu_2nd - 2 * self.expected_mu * x
        beta = self.b_0 + 0.5 * tau * (
                    self.expected_mu_2nd - self.mu_0 ** 2 - 2 * self.expected_mu * self.mu_0) + 0.5 * sum
        # beta = 0.5*sum
        self.expected_tau = self.alpha / beta
        self.lambda_n = (self.lambda_0+self.n)*self.expected_tau
        return beta

    def update_mu_elbo(self, beta):
        print("expected_mu_2nd", self.expected_mu_2nd)
        tau = (self.lambda_0 + self.n) * self.alpha / beta
        return tau

    def fit(self, tau):
        self.initialize(tau)
        old_tau = tau
        diff_tau = 1000
        diff_beta = 1000
        old_beta = 0
        print(self.actual_mu, self.actual_variance)
        while diff_tau and diff_beta > 10e-5:
            new_beta = self.update_tau_elbo(old_tau)
            new_tau = self.update_mu_elbo(new_beta)
            print("beta", new_beta)
            print("1/tau", 1 / new_tau)
            print("1/expected tau", 1 / self.expected_tau)
            plot_prior(self.actual_mu, self.actual_tau, self.actual_lambda, self.actual_a, self.actual_b, self.data)
            plt.subplot()
            plot_prior(self.mu, new_tau, self.lambda_n, self.alpha, new_beta, self.data, color='green')
            plt.show()
            diff_tau = abs(new_tau - old_tau)
            diff_beta = abs(new_beta - old_beta)
            print("diff_tau", diff_tau)
            print("diff_beta", diff_beta)
            old_beta = new_beta
            old_tau = new_tau
        return self.mu, new_tau

    def plot_normal_gamma(self, tau, beta):
        # TO DO does this work??

        mu_plot = np.linspace(-1, +1, 100)
        tau_plot = np.linspace(0, 2, 100)
        grid = np.meshgrid(mu_plot, tau_plot)
        pdf_array = np.zeros((100, 100))
        for i in range(pdf_array.shape[0]):
            for j in range(pdf_array.shape[0]):
                pdf_array[i, j] = self.normal_gamma_pdf(mu_plot[i], tau_plot[j], tau, beta)
        plt.contour(mu_plot, tau_plot, pdf_array)
        plt.show()

    #def plot_inferred_posterior(self):



def compute_normal_prob(x, mean, precision):
    dist = norm.pdf(x, mean, math.sqrt(1/precision))
    return dist

def compute_gamma_prob(tau, a, b):
    dist = gamma.pdf(tau, a, loc=0, scale=(1/b))
    return dist


def plot_prior(mu, tau, lambda_, a_n, b_n, data, color='blue'):
    #true_tau = 1 / true_variance
    #true_tau = true_variance
    N = len(data)
    mu_range = np.linspace(-1, 1, 100)
    tau_range = np.linspace(0, 2, 100)
    mu_grid, tau_grid = np.meshgrid(mu_range, tau_range, indexing='ij')
    z = np.zeros_like(mu_grid)

    for i, mu_ in enumerate(mu_range):
        for j, tau_ in enumerate(tau_range):
            #z[i, j] = normal_gamma(data, mu, tau)
            z[i, j] = compute_normal_prob(mu_, mu, lambda_*tau)*compute_gamma_prob(tau_, a_n, b_n)
    plt.contour(mu_grid, tau_grid, z, color=color)
    plt.xlabel("Mean")
    plt.ylabel("Precision")
    #plt.show()


def normal_gamma(data, my_mu, my_tau):

    ##params: from WIKI P(mu, tau | data) = NormalGamma()
    lambda_0 = 0
    a_0 = 0
    b_0 = 0
    mu_0 = 0
    mean = np.mean(data)
    N = len(data)
    s = np.var(data)
    mu = 0 #(lambda_0*mu_0 + N*np.mean(data))/(lambda_0 + N)
    lambda_ = lambda_0 + N
    a = a_0 + (N/2)
    b = 0.5*np.sum([(x_i - mean)**2 for x_i in data])
    #b = b_0 + 0.5*(N*s + ((lambda_0*N)*(mean - mu_0)**2)/(lambda_0 + N))
    gamma_a = math.gamma(a)
    distribution = (((b ** a) * np.sqrt(lambda_)) / (gamma_a * np.sqrt(2 * math.pi))) * (
            my_tau ** a * 0.5) * np.exp(-b * my_tau) * np.exp(((lambda_ * my_tau) * ((my_mu - mu) ** 2)) / 2)
    return distribution


mu = 0
sigma = 1
sample = 10
np.random.seed(50)
x = np.random.normal(loc=mu, scale=sigma, size=sample)
#p_mu = norm(mu, sigma).pdf(x)
#p_tau = gamma.pdf(x, 0)
#plot_prior(mu, sigma, x)
# z = norm.pdf(*np.meshgrid(mu_plot, tau_plot), mu_plot, tau_plot)
# plt.contour(mu_plot, tau_plot, z)
# plt.show()

# plt.scatter(x, norm.pdf(x, mu, sigma))
# plt.show()
vi = VI(x, mu, sigma)
start_tau = np.random.uniform(0,1)
#start_tau = 0.4
print('start tau', start_tau)
vi.fit(start_tau)
