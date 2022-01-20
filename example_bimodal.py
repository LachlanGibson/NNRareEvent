from rare_event import *

# Create instance of the full system.
h = Generator(
    num_dims=1,                  #generator output dimensionality
    num_input_dims = 8,          #generator input dimensionality
    genHLsizes = [16,32,64],     #generator hidden layer sizes
    pdf_netHLsizes = [16,32,64], #density net hidden layer sizes
    name = "bimodal"             #name of system
    ).to(device)

# The target log-density function (up to a constant).
def bimodal_pdf(X):
    weights = [1/4, 3/4]
    means = [[-4],[0]]
    sigmas = [0.5,1]
    return simple_multimodal(X, weights, means, sigmas)

# The true CDF, for the Kolmogorov-Smirnov test
def bimodal_cdf(X):
    weights = [1/4, 3/4]
    means = [[-4],[0]]
    sigmas = [0.5,1]
    return simple_multimodal_cdf(X, weights, means, sigmas)

gamma = None # There is no performance threshold

# Optimize the kernel bandwidth for samples of size 1000.
# The large learning rate of 0.1 accelerates the calculation.
h.optimise_t(1000, lr = 0.1)

# Train the system, only updating the density network 1000 steps.
train_system(h, 1, bimodal_pdf, identity,
    bs = 1000, gen_steps = 0, pdf_steps = 1000,
	gamma = gamma, save_interval = 1, kernel_interval = 1)

# Optimize the kernel bandwidth for samples of size 10000.
h.optimise_t(10000, lr = 0.1)

# Train the system for 3000 epochs. At each epoch the generator
# is updated once and the density network is updated 10 times.
train_system(h, 3000, bimodal_pdf, identity,
    save_interval = 100, gamma = gamma,
    kernel_interval=10,bs=10000,gen_steps=1,pdf_steps=10)

# Plot a histogram of the generator outputs
h.plot1Dhist(num_samples = 100000, num_bins = 50,
             targetLL = bimodal_pdf, gamma = gamma)
plt.show()

# Perform a one sample Kolmogorov-Smirnov test
X = h.sample_X(1000)
F = bimodal_cdf(X)
print(kolmogorov_smirnov(X,F))
