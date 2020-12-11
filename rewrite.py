import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma

class VI:
    def __init__(self, data, true_mu, true_variance):
        self.data = data
        self.true_mu = true_mu
        self.true_variance = true_variance
        N = len(data)
        sample_mean = np.mean(data)
        var = np.sum((self.data-self.true_mu)**2)
        print("var", var)
        self.actual_a = 0.5 * N
        self.actual_b = 0.5 * var
        self.actual_tau = self.actual_a / self.actual_b
        self.actual_lambda = N
        self.lambda_0 = 0
        self.a_0 = 0
        self.b_0 = 0
        self.n = N #number of data points
        self.a_n = self.n/2 #a_n never changes
        self.exp_mu = sample_mean # mu_n never changes, mu and E(mu) are the same...
        print(self.exp_mu)
        self.exp_mu_2nd = 0

    def update_b_n(self, exp_tau):
        self.update_mu_2nd(exp_tau)
        return 0.5*sum([x**2 + self.exp_mu_2nd - 2*x*self.exp_mu for x in self.data])

    def update_lambda_n(self, exp_tau):
        return (self.lambda_0 + self.n)*exp_tau

    def update_exp_tau(self, b_n):
        return self.a_n/b_n

    def update_mu_2nd(self, exp_tau):
        self.exp_mu_2nd = self.exp_mu**2 + 1/self.n*exp_tau
        #print("exp_mu_2nd", self.exp_mu_2nd)

    def compute_elbo(self, lambda_n, b_n):
        return 0.5*np.log(1/lambda_n) + np.log(math.gamma(self.a_n)) - self.a_n*np.log(b_n)


    def fit(self, exp_tau):
        diff_elbo = 1000
        old_elbo = 0
        iter = 0
        while diff_elbo > 10e-3:
            lambda_n = self.update_lambda_n(exp_tau)
            b_n = self.update_b_n(exp_tau)
            #plot_posterior(self.true_mu, self.actual_tau, self.actual_lambda, self.actual_a, self.actual_b,
             #              color='blue')
            #plot_posterior(self.exp_mu, exp_tau, lambda_n, self.a_n, b_n, color='red')
            #plt.title("after beta update %d" % iter)
            #plt.show()
            exp_tau = self.update_exp_tau(b_n)
            elbo = self.compute_elbo(lambda_n, b_n)
            print("elbo", elbo)
            diff_elbo = abs(old_elbo-elbo)
            old_elbo = elbo
            iter += 1

        print("data to plot functions: mu %f tau %f lambda %f a %f b %f" % (self.true_mu, self.actual_tau, self.actual_lambda, self.actual_a, self.actual_b))
        print("estimated data to plot functions: mu %f tau %f lambda %f a %f b %f" % (self.exp_mu, exp_tau, lambda_n, self.a_n, b_n))
        print("converged after %d iterations" % iter)
        #plot_posterior(self.true_mu, self.actual_tau, self.actual_lambda, self.actual_a, self.actual_b, color='blue')
        #plot_posterior(self.exp_mu, exp_tau, lambda_n, self.a_n, b_n, color='red')
        #plt.title("after convergence")
        #plt.show()


def compute_normal_prob(x, mean, precision):
    dist = norm.pdf(x, mean, np.sqrt(1/precision))
    return dist


def compute_gamma_prob(tau, a, b):
    dist = gamma.pdf(tau, a, loc=0, scale=(1/b))
    return dist


def plot_posterior(mu, tau, lambda_, a_n, b_n, color='blue'):
    mu_range = np.linspace(1, 3, 100)
    tau_range = np.linspace(2, 5, 100)
    mu_grid, tau_grid = np.meshgrid(mu_range, tau_range, indexing='ij')
    z = np.zeros_like(mu_grid)

    for i, mu_ in enumerate(mu_range):
        for j, tau_ in enumerate(tau_range):
            z[i, j] = compute_normal_prob(mu_, mu, lambda_*tau)*compute_gamma_prob(tau_, a_n, b_n)
    plt.contour(mu_grid, tau_grid, z, colors=color)
    plt.title("Posterior distributions blue: true posterior, red: inferred posterior")
    #plt.legend()
    plt.xlabel("Mean")
    plt.ylabel("Precision")




mu = 2
sigma = 4
sample = 110
np.random.seed(50)
x = np.random.normal(loc=mu, scale=sigma, size=sample)
#x = np.array([-1, -0.5, 0, 0.5, 1])
#guess_exp_tau = 1
vi = VI(x, mu, sigma)
guess_exp_tau = np.random.uniform(0, 1)

vi.fit(guess_exp_tau)
