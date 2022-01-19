from rare_event import *

# Create instance of the full system. Note the generator outputs
# are forced to be positive by the exp activation function.
h = Generator(
    num_dims=2,                  #generator output dimensionality
    num_input_dims = 8,          #generator input dimensionality
    genHLsizes = [16,32,64],     #generator hidden layer sizes
    pdf_netHLsizes = [16,32,64], #density net hidden layer sizes
    generator_activation = torch.exp, #generator activation func
    name = "exponential_sum_gamma10"  #name of system
    ).to(device)

gamma = 10 # The performance threshold

# The target log-density function (up to a constant).
def exponential(X):
    return -X.sum(1, keepdim = True)

# The sample performance function
def S(X):
    return X.sum(1, keepdim = True)

# Optimize the kernel bandwidth for samples of size 1000.
# The large learning rate of 0.1 accelerates the calculation.
h.optimise_t(1000, lr = 0.1)

# Train the system, only updating the density network 1000 steps.
train_system(h,1,exponential,S,gen_steps=0,pdf_steps=1000,
	gamma=gamma,save_interval=1,kernel_interval=1,bs=1000)

# Optimize the kernel bandwidth for samples of size 10000.
h.optimise_t(10000, lr = 0.1)

# Train the system for 3000 epochs. At each epoch the generator
# is updated once and the density network is updated 10 times.
train_system(h,3000,exponential,S,gen_steps=1,pdf_steps=10,
	gamma=gamma,save_interval=100,kernel_interval=10,bs=10000)

# Lower the learning rate of the generator optimizer from the
# default of 0.001 to 0.0001.
h.generator.optimiser.param_groups[0]["lr"] = 0.0003

# Train the system for another 3000 epochs.
train_system(h,3000,exponential,S,gen_steps=1,pdf_steps=10,
	gamma=gamma,save_interval=100,kernel_interval=10,bs=10000)

train_system(h,1000,exponential,S,gen_steps=1,pdf_steps=10,
	gamma=gamma,save_interval=100,kernel_interval=10,bs=10000)

# Plot a contour plot
h.plot2Dcontour(xrange = [0,14], xn = 1000, levels = 100)
plt.show()