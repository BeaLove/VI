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