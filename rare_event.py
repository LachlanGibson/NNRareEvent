import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
import os


# set device to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# =============================================================================
# BEGIN: Generator Class and Functions


class MLP(nn.Module):
    # multilayer perceptron
    def __init__(self,
                 insize,                 # The number of input nodes
                 outsize,                # The number of output nodes
                 HLsizes = [],           # A list of hidden layer sizes
                 HLact = nn.ReLU(),      # Hidden layer activation function
                 outact = nn.Identity(), # Output layer activation function
                 lr = 0.001              # Optimiser learning rate
                ):
        super().__init__()
        self.insize = insize
        self.outsize = outsize
        self.HLsizes = HLsizes
        self.HLact = HLact
        self.outact = outact
        self.lr = lr
        self.loss_history = list()
        
        sizes = [insize] + HLsizes + [outsize]
        n_layer = len(sizes) - 1
        self.linears = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i+1]) for i in range(n_layer)])
        
        self.optimiser = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=0.0001)
    
    def forward(self, x):
        for L in self.linears[:-1]:
            x = self.HLact(L(x))
        x = self.linears[-1](x)
        x = self.outact(x)
        return x
    
    def save_loss_history(self, path):
        np.savetxt(path, self.loss_history, delimiter=',')
    
    def param_requires_grad(self, *args, **kwargs):
        param_requires_grad(self, *args, **kwargs)


class Generator(nn.Module):
    def __init__(self,
                 num_dims = 1,
                 num_input_dims = 1,
                 genHLsizes = [],
                 pdf_netHLsizes = [],
                 target_netHLsizes = [],
                 S_netHLsizes = [],
                 name = "generator",
                 generator_activation = nn.Identity()
                ):
        super().__init__()
        
        self.name = name
        
        self.num_dims = num_dims
        self.num_input_dims = num_input_dims
        
        self.S_net_criterion = nn.MSELoss()
        self.target_net_criterion = nn.MSELoss()
        self.pdf_net_criterion = nn.MSELoss()
        self.generator_criterion = KLDivLoss
        
        self.epoch = 0
        
        self.t = None
        self.cov = None
        self.eye = torch.eye(num_dims).to(device)
        
        # generator network
        # input tensor size [number of samples, number of dimensions]
        # outputs a tensor of the same size
        self.generator = MLP(num_input_dims,num_dims,HLsizes = genHLsizes,
                             lr = 0.001, outact = generator_activation)
        
        # takes input from the generator output and outputs the log probability
        # density (or log-liklihood)
        # output shape is [number of samples, 1]
        self.pdf_net = MLP(num_dims,1,HLsizes = pdf_netHLsizes,lr = 0.001)
        
        # takes input from the generator output and outputs an estimate of the
        # log probability density (or log-liklihood)
        # from the target distribution
        # output shape is [number of samples, 1]
        self.target_net = MLP(num_dims,1,HLsizes=target_netHLsizes,lr = 0.001)
        
        # takes input from the generator output and outputs an estimate of the
        # sample performance function S(X)
        # output shape is [number of samples, 1]
        self.S_net = MLP(num_dims, 1, HLsizes = S_netHLsizes, lr = 0.001)
    
    def forward(self, x):
        x = self.generator(x)
        x = self.pdf_net(x)
        return x
    
    def pdf(self, x):
        # adds an exponential to the final layer to represent the probablilty
        #density (or likelihood)
        x = self.forward(x)
        x = torch.exp(x)
        return x
    
    def save_state_dict(self, f):
        """Saves the state dictionary to file.
        

        Parameters
        ----------
        f : str or other file like object
            The file name.

        Returns
        -------
        None.

        """
        torch.save(self.state_dict(), f)
    
    def load_state_dict(self, f):
        """Loads a state dictionary from file.

        Parameters
        ----------
        f : str or other file like object
            The file name.

        Returns
        -------
        None.

        """
        self.load_state_dict(torch.load(f))
    
    def save_full(self, f):
        """Saves the full object using torch.save
        
        Load the object by using torch.load(f)

        Parameters
        ----------
        f : str or other file like object
            The file name.

        Returns
        -------
        None.

        """
        torch.save(self, f)
    
    def optimise_t(self, num_samples, **kwargs):
        """Use optimal_width to optimise t.
        
        See optimal_width for details.

        Parameters
        ----------
        num_samples : int
            The number of samples.
        **kwargs : keyword arguments
            See optimal_width for details.

        Returns
        -------
        None.

        """
        with torch.no_grad():
            X = self.sample_X(num_samples)
            
            self.t = optimal_width(X, t0 = self.t, Xcov = "scalar", **kwargs)
            #self.cov = X.T.cov()*self.t
            self.cov = self.t*self.eye*X.var(0).mean()
    
    def sample_input(self, *args, **kwargs):
        return sample_input(self, *args, **kwargs)
    
    def sample_X(self, *args, **kwargs):
        return sample_X(self, *args, **kwargs)
    
    def param_requires_grad(self, *args, **kwargs):
        param_requires_grad(self, *args, **kwargs)
    
    def evaluate(self, *args, **kwargs):
        return evaluate(self, *args, **kwargs)
    
    def estimate_norm(self, *args, **kwargs):
        return estimate_norm(self, *args, **kwargs)
    
    def train_S_net(self, *args, **kwargs):
        train_S_net(self, *args, **kwargs)
    
    def train_target_net(self, *args, **kwargs):
        train_target_net(self, *args, **kwargs)
    
    def train_pdf_net(self, *args, **kwargs):
        train_pdf_net(self, *args, **kwargs)
    
    def train_generator(self, *args, **kwargs):
        train_generator(self, *args, **kwargs)
    
    def train_system(self, *args, **kwargs):
        train_system(self, *args, **kwargs)
    
    def plot1Dhist(self, *args, **kwargs):
        plot1Dhist(self, *args, **kwargs)
    
    def plot2Dcontour(self, *args, **kwargs):
        plot2Dcontour(self, *args, **kwargs)


def sample_input(g, num_samples, num_dims = None):
    """Returns a random tensor appropriate for input into a generator.
    
    Parameters
    ----------
    g : Generator
        The generator.
    num_samples : int
        The number of samples. Should be a positive integer.
    num_dims : int, optional
        The number of dimensions. The default is self.num_input_dims.

    Returns
    -------
    torch.Tensor of shape [num_samples, num_dims]
        A tensor of standard normal random numbers.

    """
    if num_dims is None:
        num_dims = g.num_input_dims
    return torch.randn([num_samples, num_dims]).to(device)


def sample_X(g, num_samples):
    """Samples a generator.
    
    Returns a tensor of samples produced by the generator. Note that
    .eval() and torch.no_grad() are not called within this function.

    Parameters
    ----------
    g : Generator
        The generator.
    num_samples : int
        The number of samples. Should be a positive integer.

    Returns
    -------
    X : torch.Tensor of shape [num_samples, self.num_dims]
        A tensor of samples from the generator.

    """
    r = g.sample_input(num_samples)
    X = g.generator(r)
    return X


def param_requires_grad(n, TF):
    """Sets the requires_grad values of the parameters in a network.

    Parameters
    ----------
    n : nn.Module
        The network.
    TF : bool
        The value (True or False).

    Returns
    -------
    None.

    """
    for param in n.parameters():
        param.requires_grad = TF


def evaluate(g, targetLL, S, num_samples = 1000,
             alpha = 100, gamma = None, append_loss = False, cov = None):
    """Evaluates the networks in an instance of Generator.
    

    Parameters
    ----------
    g : Generator
        The generator.
    targetLL : function
        The target function which computes the target log-liklihood from an X
        tensor of shape [num_samples, num dims].
    S : function
        Sample performance function.
    num_samples : int, optional
        DESCRIPTION. The default is 1000.
    alpha : float or int, optional
        The penalty scale alpha*(gamma-S(X)). The default is 100.
    gamma : float or int, optional
        When not None, will modify the target with S(X) < gamma.
        The default is None.
    append_loss : bool, optional
        Appends loss values to the loss_history of each network when True.
        The default is False.
    cov : torch.Tensor of shape [num dims, num dims], optional
        Covariance matrix of the Gaussian kernel. The default is computed from
        the data using Scott's rule.

    Returns
    -------
    gen_loss : float
        generator loss.
    pdf_loss : float
        pdf_net loss.

    """
    g.eval()
    with torch.no_grad():
        X = g.sample_X(num_samples)
        V = g.pdf_net(X)
        ks = ksdensity(X, cov = cov).log().view(-1,1)
        pdf_loss = g.pdf_net_criterion(V, ks).item()
        
        target = targetLL(X)
        if gamma is not None:
            target = target - perf_penalty(S(X), alpha=alpha, gamma=gamma)
        gen_loss = g.generator_criterion(V, target).item()
        
        if append_loss:
            g.generator.loss_history.append(gen_loss)
            g.pdf_net.loss_history.append(pdf_loss)
        
        return (gen_loss, pdf_loss)


def estimate_norm(g, num_samples = 1000, targetLL = None, S=None, gamma=None,
                  pdf_net = True, cov = None):
    """Estimates the normalisation constant of a target distribution.
    
    
    Parameters
    ----------
    g : Generator
        The generator.
    num_samples : int, optional
        The number of samples. The default is 1000.
    targetLL : function, optional
        The target function which computes the target log-liklihood from an X
        tensor of shape [num_samples, num dims]. The default is ones.
    S : function, optional
        Sample performance function. The default is the identity function.
    gamma : float or int, optional
        When not None, will modify the target with S(X) < gamma.
        The default is None.
    pdf_net : bool, optional
        Use g.pdf_net to estimate the pdf when True, otherwise use ksdensity.
        The default is True.
    cov : torch.Tensor of shape [num dims, num dims], optional
        Covariance matrix of the Gaussian kernel. The default is None.

    Returns
    -------
    me : float
        Estimate of normalisation constant.
    se : float
        Standard error of normalisation estimate.

    """
    
    if S is None:
        S = identity
    
    g.eval()
    with torch.no_grad():
        X = g.sample_X(num_samples)
        if pdf_net:
            N = g.pdf_net(X)
        else:
            N = ksdensity(X, cov = cov).log()
        if targetLL is None:
            f = 0
        else:
            f = targetLL(X)
        
        if gamma is None:
            thresh = 1
        else:
            thresh = S(X) >= gamma
        
        summand = thresh*(f-N).exp()
        se = summand.std().item()/np.sqrt(num_samples)
        me = summand.mean().item()
        return me, se


# END:   Generator Class and Functions
# =============================================================================
# BEGIN: Training functions


def train_S_net(g, steps, S, bs = 1000):
    """Trains the S net.
    
    At each iteration the generator generates "bs" samples. At each sample the
    sample performance is estimated using g.S_net and compared to the
    values given by S. The g.S_net_criterion is then minimised.

    Parameters
    ----------
    g : Generator
        The generator.
    steps : int
        The number of training steps.
    S : function
        The sample performance function which takes input of X tensor
        of shape [num_samples, number of dims]. The default is None.
    bs : int, optional
        Batch size of each training step. The default is 1000.

    Returns
    -------
    None.

    """
    
    g.S_net.train()
    g.S_net.param_requires_grad(True)
    for i in range(steps):
        g.S_net.optimiser.zero_grad()
        X = g.sample_X(bs)
        Y = g.S_net(X)
        
        with torch.no_grad():
            s = S(X)
        
        S_loss = g.S_net_criterion(Y,s)
        S_loss.backward()
        g.S_net.optimiser.step()


def train_target_net(g, steps, targetLL, bs = 1000):
    """Trains the target net.
    
    At each iteration the generator generates "bs" samples. At each sample the
    target log likelihood is estimated using g.target_net and compared to the
    actual target. The g.target_net_criterion is then minimised.
    
    The training steps roughly follow...
    
    for steps times:
        Sample the generator network with bs samples.
        Estimate the target log-likelihood of those samples using g.target_net.
        Calculate the true target log-likelihood using targetLL.
        Estimate the loss between estimated and true target log-likelihood
            values using g.target_net_criterion (such as MSE).
        Update target_net network parameters using gradient descent.
    
    * Note that gradients are not backpropagated through targetLL.
    
    Parameters
    ----------
    g : Generator
        The generator.
    steps : int
        The number of training steps.
    targetLL : function
        The target function which computes the target log-liklihood from an X
        tensor of shape [num_samples, number of dims]. The default is None.
    bs : int, optional
        Batch size of each training step. The default is 1000.

    Returns
    -------
    None.

    """
    
    g.target_net.train()
    g.target_net.param_requires_grad(True)
    for i in range(steps):
        g.target_net.optimiser.zero_grad()
        X = g.sample_X(bs)
        Y = g.target_net(X)
        
        with torch.no_grad():
            target = targetLL(X)
        
        target_loss = g.target_net_criterion(Y,target)
        target_loss.backward()
        g.target_net.optimiser.step()


def train_pdf_net(g, steps, bs = 1000, cov = None):
    """Trains the pdf net.
    
    At each iteration the generator generates "bs" samples. At each sample the
    probability density is estimated using the "ksdensity" which becomes the
    target for the output of the pdf_net. The g.pdf_net_criterion is then
    minimised.
    
    The training steps roughly follow...
    
    for steps times:
        Sample the generator network with bs samples.
        Estimate the gnerator log-pdf of those samples using g.pdf_net.
        Estimate the target log-pdf of those samples using ksdensity.
        Estimate the loss between estimated log-pdf and target log-pdf
            values using g.pdf_net_criterion (such as MSE).
        Update pdf_net network parameters using gradient descent.
    
    * Note that gradients are not backpropagated through the kernel density
        estimation.

    Parameters
    ----------
    g : Generator
        The generator.
    steps : int
        The number of training steps.
    bs : int, optional
        Batch size of each training step. The default is 1000.
    cov : torch.Tensor of shape [num dims, num dims], optional
        Covariance matrix of the Gaussian kernel. The default is computed from
        the data using Scott's rule.

    Returns
    -------
    None.

    """
    g.generator.eval()
    g.generator.param_requires_grad(False)
    g.pdf_net.train()
    g.pdf_net.param_requires_grad(True)
    for i in range(steps):
        g.pdf_net.optimiser.zero_grad()
        X = g.sample_X(bs)
        V = g.pdf_net(X)

        with torch.no_grad():
            ks = ksdensity(X, cov = cov).log().view(-1,1)
        pdf_loss = g.pdf_net_criterion(V,ks)
        pdf_loss.backward()
        g.pdf_net.optimiser.step()


def train_generator(
        g, steps, bs = 1000,
        targetLL = None, S = None,
        gamma = None, alpha = 100):
    """Trains the generator network.
    
    The training steps roughly follow...
    
    for steps times:
        Sample the generator network with bs samples.
        Estimate the gnerator log-pdf of those samples using g.pdf_net.
        Compute/estimate the target log-likelihood of those samples using
            targetLL or g.target_net.
        Compute/estimate sample performance of samples using S or g.S_net
            if gamma is not None.
        Penalise target log-likelihood of samples whose performace is less than
            gamma if gamma is not None.
        Estimate the loss between estimated log-pdf and target log-likelihood
            values using g.generator_criterion (such as KL divergence).
        Update generator network parameters using gradient descent.
    
    * Note that targetLL and S need to be differentiable so that gradients can
        be backpropagated.
    
    Parameters
    ----------
    g : Generator
        The generator.
    steps : int
        The number of training steps.
    bs : int, optional
        Batch size of each training step. The default is 1000.
    targetLL : function
        The target function which computes the target log-liklihood from an X
        tensor of shape [num_samples, number of dims]. The default is None.
    S : function, optional
        Sample performance function. The default is g.S_net.
    gamma : float or int, optional
        When not None, will modify the target with S(X) < gamma.
        The default is None.
    alpha : float or int, optional
        The penalty scale alpha*(gamma-S(X)). The default is 100.

    Returns
    -------
    None.

    """
    g.generator.train()
    g.generator.param_requires_grad(True)
    g.pdf_net.eval()
    g.pdf_net.param_requires_grad(False)
    if targetLL is None:
        g.target_net.eval()
        g.target_net.param_requires_grad(False)
    if S is None:
        g.S_net.eval()
        g.S_net.param_requires_grad(False)
        
    for i in range(steps):
        g.generator.optimiser.zero_grad()
        X = g.sample_X(bs)
        V = g.pdf_net(X)
        
        if targetLL is None:
            target = g.target_net(X)
        else:
            target = targetLL(X)
        
        if gamma is not None:
            if S is None:
                sperf = g.S_net(X)
            else:
                sperf = S(X)
            target = target - perf_penalty(sperf, alpha=alpha, gamma=gamma)
        
        gen_loss = g.generator_criterion(V, target)
        gen_loss.backward()
        g.generator.optimiser.step()


def train_system(g, nepochs, targetLL, S,
                    bs = 1000, gen_bs = None, pdf_bs = None,
                    gen_steps = 1, pdf_steps = 1,
                    print_interval = 10, gamma = None, alpha = 100,
                    save_interval = None, append_loss = True,
                    exp_path = "experiments", exp_name = None,
                    eval_samples = 10000, kernel_interval = None):
    """Trains the full generator system.
    
    The training steps roughly follow...
    
    for nepochs times:
        Train generator network for gen_steps with batch-sizes of gen_bs.
        Train pdf network for pdf_steps with batch-sizes of pdf_bs.
        Evaluate networks using eval_samples number of samples.
        If at a save interval, save the generator to file.
        If at a print interval, print the evaluated loss values.
    

    Parameters
    ----------
    g : Generator
        The generator.
    nepochs : int
        The number of epochs. The number of training iterations.
    targetLL : function
        The target function which computes the target log-liklihood from an X
        tensor of shape [num_samples, number of dims].
    S : function
        Sample performance function. The input and output are tensors of the
        same shapes as for targetLL.
    bs : int, optional
        The batch size for network training. This is the default value for
        gen_bs. The default is 1000.
    gen_bs : int, optional
        Generator network training batch size. The default is bs.
    pdf_bs : int, optional
        PDF network training batch size. The default is gen_bs.
    gen_steps : int, optional
        The number of generator network training steps each epoch.
        The default is 1.
    pdf_steps : int, optional
        The number of PDF net training steps each epoch. The default is 1.
    print_interval : int, optional
        The evaluated losses are printed every print_interval of epochs.
        The default is 10.
    gamma : float or int, optional
        When not None, will modify the target with S(X) < gamma.
        The default is None.
    alpha : float or int, optional
        The penalty scale alpha*(gamma-S(X)). The default is 100.
    save_interval : int, optional
        The system (g) is saved to file every save_interval of epochs.
        The default is None (meaning it is never saved).
    append_loss : bool, optional
        Appends loss values to the loss_history of each network during the
        evaluation step when True.
        The default is True.
    exp_path : str or other path type, optional
        Path to experiments. The path will be created if it does not already
        exist and save_interval is not None. The default is "experiments".
    exp_name : str, optional
        The experiment name. A folder with this name in exp_path will be
        created/used to save the system if save_interval is not None.
        The default is g.name.
    eval_samples : int, optional
        The sample size used to evaluate the networks. The default is 10000.
    kernel_interval : int, optional
        The kernel width will be optimised every kernel_interval epochs.
        The default is None.

    Returns
    -------
    None.

    """
    
    
    if exp_name is None:
        exp_name = g.name
    
    if save_interval is not None:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        path = os.path.join(exp_path, exp_name)
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, "checkpoints")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
    
    if gen_bs is None:
        gen_bs = bs
    if pdf_bs is None:
        pdf_bs = gen_bs
    
    if kernel_interval is not None:
        cov = None
    
    for i in range(nepochs):
        
        g.train_generator(gen_steps, bs = gen_bs,
            targetLL = targetLL, S = S, gamma = gamma, alpha = alpha)
        
        
        if kernel_interval is not None:
            if i%kernel_interval==0:
                g.optimise_t(pdf_bs)
                cov = g.cov
        
        g.train_pdf_net(pdf_steps, bs = pdf_bs, cov = cov)
        
        
        loss = g.evaluate(
            targetLL, S, num_samples = eval_samples,
            alpha = alpha, gamma = gamma, append_loss = append_loss, cov = cov)
        
        if save_interval is not None:
            if i%save_interval==(save_interval-1):
                f = os.path.join(checkpoint_path, "epoch"+str(g.epoch)+".pt")
                g.save_full(f)
        
        if print_interval is not None:
            if i%print_interval==(print_interval-1):
                print(
                    i+1,
                    'gen_loss: %.8f, pdf_loss: %.8f' % loss)#, end ='\r')
        
        g.epoch += 1


# END:   Training functions
# =============================================================================
# BEGIN: Plots


def plot1Dhist(g, num_samples = 1000, num_bins = None, targetLL = None,
               renorm = False, gamma = None, alpha = 100, S = None,
               hide_bins = False, loc = "best", factor = None):
    """Plots the generator histogram along the first dimension.
    
    The network estimated pdf and target pdf are also plotted.

    Parameters
    ----------
    g : Generator
        The generator.
    num_samples : int, optional
        The number of samples to produce the histogram. The default is 1000.
    num_bins : int, optional
        The number of bins in the histogram. The default is num_samples//20.
    targetLL : function, optional
        A function which computes the target log-liklihood from an X tensor
        of shape [num_samples, number of dims]. Not plotted when None.
        The default is None.
    renorm : bool, optional
        When True, renormalises the target pdf so that the area in the sampled
        domain is 1. The default is False.
    gamma : float or int, optional
        When not None, will modify the target with S(X) < gamma.
        The default is None.
    alpha : float or int, optional
        The penalty scale (exp(alpha*(gamma-S(X)))). The default is 100.
    S : function, optional
        Sample performance function. The default is the identity function.
    hide_bins : bool, optional
        Does not plot the histogram when True. The default is False.
    loc : str, optional
        Legend location. The default is "best".
    factor : float, optional
        Normalisation factor. The default is computed using trapezoid rule.

    Returns
    -------
    None.

    """
    if S is None:
        S = identity
    if num_bins is None:
        num_bins = num_samples//20
    with torch.no_grad():
        g.eval()
        X = g.sample_X(num_samples)
        X = X[X[:,0].sort().indices,:]
        V = g.pdf_net(X).exp()
    if not hide_bins:
        plt.hist(X[:,0].tolist(),bins=num_bins,density=True,label="Histogram")
    plt.plot(X[:,0].tolist(), V[:,0].tolist(),'r', label = "Learned Density")
    if targetLL is not None:
        t = targetLL(X).exp()
        if gamma is not None:
            t = t/perf_penalty(S(X), alpha = alpha, gamma = gamma).exp()
        if renorm:
            if factor is None:
                factor = ((t[1:,0]+t[:-1,0])*(X[1:,0]-X[:-1,0])).sum()/2
            t = t/factor
        plt.plot(X[:,0].tolist(),t[:,0].tolist(),'b--',label="Target Density")
    plt.xlabel("X")
    plt.ylabel("Density")
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.legend(loc = loc)


def plot2Dcontour(g, xrange = [-2,2], yrange = None, num_samples = 1000,
                  xn = 100, yn = None, levels = 20, colourbar = True):
    """Plots a contour plot with an overlaid scatter plot of a random sample.
    
    Parameters
    ----------
    g : Generator
        The generator.
    xrange : tuple or list of length 2, optional
        Range of x grid values. The default is [-2,2].
    yrange : tuple or list of length 2, optional
        Range of y grid values. The default is xrange.
    num_samples : int, optional
        The number of samples to overlay the contour plot. The default is 1000.
    xn : int, optional
        The number of x grid values for computing the contours.
        The default is 100.
    yn : int, optional
        The number of y grid values for computing the contours.
        The default is xn.
    levels : int, optional
        The number of contours. The default is 20.
    colourbar : bool, optional
        Includes a colour bar when True. The default is True.

    Returns
    -------
    None.

    """
    if yrange is None:
        yrange = xrange
    if yn is None:
        yn = xn
    x = torch.linspace(xrange[0], xrange[1], xn, requires_grad=False)
    y = torch.linspace(yrange[0], yrange[1], yn, requires_grad=False)
    
    X, Y = torch.meshgrid(x, y)
    
    XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim = 1).to(device)
    
    with torch.no_grad():
        g.eval()
        P = g.pdf_net(XY).exp().reshape(X.shape)
    
    S = g.sample_X(num_samples).to("cpu")
    cs = plt.contourf(
        X.to("cpu").numpy(), Y.to("cpu").numpy(), P.to("cpu").numpy(),
        cmap=cm.PuBu, levels = levels, zorder=-4)
    ax = plt.gca()
    ax.plot(S[:,0].tolist(),S[:,1].tolist(),"r.",alpha = 1, markersize = 0.5)
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_aspect(1./ax.get_data_ratio())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_rasterization_zorder(-1)
    if colourbar:
        plt.gcf().colorbar(cs)


# END:   Plots
# =============================================================================
# BEGIN: Distributions


def ksdensity(x, y = None, cov = None, power_scale = 1, CV = False):
    """Kernel density estimation using multivariate normal kernel.
    

    Parameters
    ----------
    x : torch.Tensor of shape [num samples, num dims]
        Sample points used to estimate the PDF.
    y : torch.Tensor of shape [num locations, num dims], optional
        Locations to compute the estimated PDF. The default is x.
    cov : torch.Tensor of shape [num dims, num dims], optional
        Covariance matrix of the Gaussian kernel. The default is computed from
        the data using Scott's rule.
    power_scale : float or int, optional
        How the kenel covariance scales with the number of points.
        num samples^(-2 * power_scale/(4 + num dims)).
        The default is 1, which represents Scott's Rule.
    CV : bool, optional
        When True and y is None, will estimate the pdf at each sample without
        including the kernel at that sample. y must be None when CV is True.
        The default is False.

    Returns
    -------
    z : torch.Tensor of shape [num locations]
        The estimated probability density value at each location.

    """
    N, K = x.shape
    
    if CV and y is not None:
        raise Exception("y should be None when CV is True")
    
    if y is None:
        y = x
    
    if cov is None:
        with torch.no_grad():
            cov = torch.cov(x.T)*(N**(-2*power_scale/(K+4)))
    
    z = y.unsqueeze(0) - x.unsqueeze(1)
    if cov.numel() == 1:
        z = -(z*z/cov).sum(2)
        z = torch.sqrt(z.exp()/(2*np.pi*cov))
    else:
        with torch.no_grad():
            cov_inv = torch.inverse(cov)
        z = -((z @ cov_inv) * z).sum(2)
        z = torch.sqrt(z.exp()/(2*np.pi)**K/cov.det())
    
    if CV and (z.shape[0] == z.shape[1]):
        z.fill_diagonal_(0)
        z = z.mean(0)*N/(N-1)
    else:
        z = z.mean(0)
    return z


def ksdensity_g(X, cov):
    """Modified estimate of the integrated squared error.
    
    The integrated squared error (ISE) is the integral over the squared
    difference between the kernel density estimate and the true density.
    Minimising the ISE (with respect to the bandwidth) is equivalent to
    minimising this function, which omits the expected pdf term (which does
    not depend on the kernel).

    Parameters
    ----------
    X : torch.Tensor of shape [num samples, num dims]
        The samples.
    cov : torch.Tensor of shape [num dims, num dims]
        Covariance matrix of the Gaussian kernel.

    Returns
    -------
    float
        An estimate of the ISE subtract the expected pdf.

    """
    with torch.no_grad():
        g = ksdensity(X, cov = 2*cov)-2*ksdensity(X, cov = cov, CV = True)
        return g.mean().item()


def optimal_width(X, lr = 0.01, Xcov = "cov", t0 = None, max_iterations = 100):
    """Identify the optimal bandwidth using Least Squares Cross Validation.
    
    Assuming standard multivariate normal kernels with covariance cov(X)*t,
    this function uses a form of gradient descent to estimate the value
    of t that minimises the integrated squared error, using ksdensity_g.

    Parameters
    ----------
    X : torch.Tensor of shape [num samples, num dims]
        The samples.
    lr : float, optional
        The learning rate. Must satisfy 0<lr<1. The default is 0.01.
    Xcov : torch.Tensor of shape [num dims, num dims] or str, optional
        Covariance matrix of the samples. Covariance of kernel is t*Xcov.
        If 'cov', then will be set to the covariance of the data. If 'diag',
        then will be set to the diagonal of the covariance matrix. If any other
        string, then will be set to the mean variance times the identity
        matrix. The default is 'cov'.
    t0 : float, optional
        Initial estimate of t. If None is provided then Scott's rule is used,
        t0 = num_samples**(-2/(num_dims+4)). The default is None.
    max_iterations : int, optional
        The maximum number of iterations. t0 can be scaled up to a factor of
        (1+-lr)^max_iterations, which is about e when max_iterations is about
        1/lr. The default is 100.

    Returns
    -------
    t0 : float
        An estimate of t which minimises the integrated squared error using
        multivariate normal kernels with covariance cov(X)*t.

    """
    if isinstance(Xcov, str):
        if Xcov == "cov":
            Xcov = torch.cov(X.T)
        elif Xcov == "diag":
            Xcov = X.var(0).diag_embed()
        else:
            Xcov = X.var(0).mean()
            Xcov = Xcov*torch.eye(X.shape[1]).to(Xcov.device)
    if t0 is None:
        N, K = X.shape
        t0 = N**(-2/(K+4))
    
    t1 = t0*(1-lr)
    g0 = ksdensity_g(X, Xcov*t0)
    g1 = ksdensity_g(X, Xcov*t1)
    if g0 == g1:
        return t0
    
    sign0 = np.sign((g1-g0)/(t1-t0))
    sign1 = sign0
    i = 0
    while (sign0 == sign1) and (i < max_iterations):
        g0, t0 = min((g0,t0),(g1,t1))
        t1 = t0*(1 - sign0*lr)
        g1 = ksdensity_g(X, Xcov*t1)
        sign1 = np.sign((g1-g0)/(t1-t0))
        i += 1
    return t0


def simple_normal(X, sigma = 1):
    """Multi-variate normal probability density function.
    
    Returns the log-probablility density at X of a normal distribution with
    mean zero and standard deviation sigma.

    Parameters
    ----------
    X : torch.Tensor of shape [num samples, num dims]
        Locations to compute the probablility density.
    sigma : float or int, optional
        The standard deviation. The default is 1.

    Returns
    -------
    logpdf : torch.Tensor of shape [num samples, 1]
        The log-probability density at each sample in X.

    """
    
    K = X.shape[1] # number of dimensions
    logpdf = -0.5*torch.pow(X/sigma, 2).sum(1)
    logpdf = logpdf - K*torch.log(sigma*torch.sqrt(torch.tensor(2*np.pi)))
    logpdf = logpdf.view(-1,1)
    return logpdf


def simple_normal_cdf(X, sigma = 1):
    """The cumulative distribution function of a normal distribution.
    

    Parameters
    ----------
    X : torch.Tensor of shape [num samples, num dims]
        Locations to compute the probablility density. Will be converted to
        a torch.Tensor if not already.
    sigma : float or int, optional
        The standard deviation. The default is 1.

    Returns
    -------
    torch.Tensor of shape [num samples]
        The cdf at each sample in X.

    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)
    if X.ndim > 1:
        X = X[:,0]
    return (1+torch.erf(X/(sigma*np.sqrt(2))))/2


def truncated_normal_cdf(X, gamma, sigma = 1):
    """The cumulative distribution function of a truncated normal distribution.
    
    The cdf for X < gamma is zero.
    

    Parameters
    ----------
    X : torch.Tensor of shape [num samples, num dims]
        Locations to compute the probablility density.
    gamma : float or int
        The truncation value. The cdf for X < gamma is zero.
    sigma : float or int, optional
        The standard deviation. The default is 1.

    Returns
    -------
    cdf : torch.Tensor of shape [num samples]
        The cdf at each sample in X.

    """
    area = (1+torch.erf(torch.tensor(gamma)/(sigma*np.sqrt(2))))/2
    cdf = simple_normal_cdf(X, sigma = sigma)
    cdf = (cdf-area)/(1-area)
    cdf = cdf.clamp(0)
    return cdf


def simple_multimodal(X, weights, means, sigmas, normalise_weights = False):
    """Computes the probablility density of a multimodal distribution.
    
    The distribution is represented by a linear combination of simple normal
    distributions.

    Parameters
    ----------
    X : torch.Tensor of shape [num samples, num dims]
        Sample points used to estimate the PDF.
    weights : torch.Tensor or list of shape [num modes]
        Weighting of each Mode.
    means : torch.Tensor or list of shape [num modes, num dims]
        Means of each mode.
    sigmas : torch.Tensor or list of shape [num modes]
        Standard deviation of each mode.
    normalise_weights : bool, optional
        Normalises the weights when True. The default is False.

    Returns
    -------
    logpdf : torch.Tensor of shape [num samples, 1]
        The log-density of each sample.

    """
    
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights).to(device)
    
    if not isinstance(means, torch.Tensor):
        means = torch.tensor(means).to(device)
    
    if isinstance(sigmas, torch.Tensor):
        sigmas = sigmas.tolist()
    
    N = weights.numel()
        
    if normalise_weights:
        weights = weights/weights.sum()
    
    means = means.unsqueeze(0)
    
    pdf = torch.stack(
        [simple_normal(X-means[:,i,:], sigma = sigmas[i]) for i in range(N)])
    logpdf = (pdf.exp()*weights.view(-1,1,1)).sum(0).log().view(-1,1)
    return logpdf


def simple_multimodal_cdf(X, weights, means, sigmas, normalise_weights=False):
    """Computes the cumulative distribution function of a multimodal dist.
    
    See the 'simple_multimodal' function for the probability density.

    Parameters
    ----------
    X : torch.Tensor of shape [num samples, num dims]
        Sample points used to estimate the PDF.
    weights : torch.Tensor or list of shape [num modes]
        Weighting of each Mode.
    means : torch.Tensor or list of shape [num modes, num dims]
        Means of each mode.
    sigmas : torch.Tensor or list of shape [num modes]
        Standard deviation of each mode.
    normalise_weights : bool, optional
        Normalises the weights when True. The default is False.

    Returns
    -------
    cdf : torch.Tensor of shape [num samples, 1]
        The cumulative distribution function at each sample.

    """
    
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights).to(device)
    
    if not isinstance(means, torch.Tensor):
        means = torch.tensor(means).to(device)
    
    if isinstance(sigmas, torch.Tensor):
        sigmas = sigmas.tolist()
    
    N = weights.numel()
        
    if normalise_weights:
        weights = weights/weights.sum()
    
    means = means.unsqueeze(0)
    
    cdf = torch.stack(
        [simple_normal_cdf(X-means[:,i,:], sigma=sigmas[i]) for i in range(N)])
    cdf = (cdf*weights.view(-1,1)).sum(0)
    return cdf


def generate_exp_sum(num_samples, gamma = 0, error = 1e-14, max_iter = 100):
    """Generate samples from a truncated sum of exponentials distribution.
    
    The pdf is f(x,y) = c*exp(-x-y) when x+y>=gamma and 0 otherwise,
    where c = (gamma+1)*exp(-gamma).
    
    x+y is sampled via inverse transform sampling which solves an equation
    iteratively (has some convergence issues when gamma = 0). Then x is sampled
    using a uniform distribution between 0 and x+y.    

    Parameters
    ----------
    num_samples : int
        The number of smaples.
    gamma : float, optional
        The threshold for x+y. The default is 0.
    error : float, optional
        The maximum difference in consecutive estimates of x+y.
        The default is 1e-14.
    max_iter : int, optional
        Maximum number of iterations for solving x+y. The default is 100.

    Returns
    -------
    X : torch.Tensor of shape [num_samples, 2]
        The samples.

    """
    if gamma == 0:
        c = 1
    else:
        c = (gamma+1)*np.exp(-gamma)
    u = torch.rand(num_samples).to(device)
    z0 = torch.ones_like(u)*gamma*1.1
    z = (z0+1).log()-torch.log(c*(1-u))
    i = 1
    while ((z-z0).abs().max() >= error) and i < max_iter:
        z0 = z
        z = (z0+1).log()-torch.log(c*(1-u))
        i += 1
    x = z*torch.rand(num_samples).to(device)
    y = z - x
    X = torch.stack([x,y],-1)
    return X


def exp_sum_cdf(X, gamma = 0):
    """The cumulative distribution function of truncated sum of exponentials.
    

    Parameters
    ----------
    X : torch.Tensor of shape [num samples, 2]
        Tensor of samples.
    gamma : float, optional
        The threshold for x+y>=gamma. The default is 0.

    Returns
    -------
    F : torch.Tensor of shape [num samples]
        The theoretical cumulative distribution function.

    """
    if gamma == 0:
        c = 1
    else:
        c = (1+gamma)*np.exp(-gamma)
    F = X.sum(1).ge(gamma)/c*(
        (-X.sum(1)).exp()
        -(-X.clamp(min = gamma)).exp().sum(1)
        -np.exp(-gamma)*(gamma-X).clamp(min = 0).sum(1)
        +c
        )
    return F


# END:   Distributions
# =============================================================================
# BEGIN: Loss and other functions


def KLDivLoss(V, target):
    """An estimate of the KL divergence from target to V. DKL(V||target)
    
    The Kullback-Leibler divergence is estimated by sampling from the V
    distribution and then computing the mean difference in loglikelihoods
    V-target.
    
    The KL divergence is minimised when the two distributions are equal
    (up to a constant factor, or additive constant on the log scale). If both
    functions are normalised then the minimal divergence is zero (although
    estimates may differ depending on the sampled points).
                                                                  

    Parameters
    ----------
    V : torch.Tensor of shape [num samples, 1]
        log likelihood values from the V distribution.
    target : torch.Tensor of shape [num samples, 1]
        log likelihood values from the target distribution.

    Returns
    -------
    torch.Tensor of shape []
        The estimated KL divergence.

    """
    
    return (V-target).mean()


def identity(X):
    """The identity function which returns the single input.
    

    Parameters
    ----------
    X : Any type
        Input.

    Returns
    -------
    X : Any type
        The input is returned.

    """
    return X


def perf_penalty(x, gamma = 0, alpha = 1):
    """Performance penalty.
    
    The penalty to the log liklihood of the target distribution.
    This should be subtracted from the target log likelihood.
    The penalty is 0 when x>=gamma but is alpha*(gamma-x) when x<gamma.
    

    Parameters
    ----------
    x : torch.Tensor of any non-empty shape
        Usually x  is the sample performance S(X).
    gamma : float or int, optional
        The threshold to apply the penalty. The default is 0.
    alpha : float or int, optional
        The strength of the penalty. The decay constant of the exponential
        factor that scales the penalised likelihoods. The default is 1.

    Returns
    -------
    torch.Tensor with the same shape as x
        The penalty to the log likelihood.

    """
    #  Returns a tensor of the same size as x.
    # For values less than gamma, the penalty is alpha times the difference.
    # For values greater than or equal to gamma, the penalty is zero.
    return alpha*nn.functional.relu(gamma - x)


def empirical_cdf(X, Y = None):
    """Empirical cumulative distribution function.
    
    The empirical cumulative distribution at Y is the proportion of samples
    in X where all components Y are larger than or equal to the corresponding 
    components in X.
    
    The function is defined by: F(Y) = 1/N sum_{Y>=X}(1).

    Parameters
    ----------
    X : torch.Tensor of shape [num samples, num dims] or [num samples]
        Sample points used to estimate the ECDF.
    Y : torch.Tensor shape [num locations, num dims], [num locations], optional
        Locations to compute the ECDF. The number of dims in Y and X must be
        the same. The default is X.

    Returns
    -------
    torch.Tensor of shape [num locations]
        The ECDF at each location in Y.

    """
    if X.ndim == 1:
        X = X.unsqueeze(1)
    
    if Y is None:
        Y = X
    elif Y.ndim == 1:
        Y = Y.unsqueeze(1)
    
    return Y.unsqueeze(1).ge(X.unsqueeze(0)).prod(2,dtype=torch.float).mean(1)


def kolmogorov_cdf(x, num_terms = 10, thresh = 0.8):
    """The Kolmogorov cumulative distribution function.
    

    Parameters
    ----------
    x : float
        CDF input.
    num_terms : int, optional
        The number of terms used in the series. The default is 10.
    thresh : float, optional
        Threshold value of x to determine which series representation to be
        computed. The default is 0.8.

    Returns
    -------
    float
        cumulative density at x. Pr(K<=x)

    """
    
    k = torch.arange(1,num_terms+1)
    
    if x < thresh:
        cdf =np.sqrt(2*np.pi)/x*torch.exp(-(2*k-1).pow(2)*(np.pi/x)**2/8).sum()
    else:
        cdf = 1-2*torch.sum((-1)**(k-1)*torch.exp(-2*(x*k).pow(2)))
    
    return cdf.item()


def kolmogorov_smirnov(X, F, correction = True):
    """Kolmogorov-Smirnov goodness of fit test.
    
    Returns the probability that the maximum difference between the ECDF of the
    samples X, and F is greater than or equal to the identified difference
    under the null hypothesis that X was sampled from F.

    Parameters
    ----------
    X : torch.Tensor of shape [num samples, num dims]
        Sample points.
    F : torch.Tesnor of shape [num samples] or [num samples, 1]
        The true cumulative distribution values at each sample in X.
    correction : bool, optional
        When True a correction is applied from Vrbik, Jan (2018). "Small-Sample
        Corrections to Kolmogorov-Smirnov Test Statistic". Pioneer Journal of
        Theoretical and Applied Statistics. 15 (1-2): 15-23.
        The default is True.

    Returns
    -------
    float
        p-value. Probability that the maximum distance between the empirical
        and true CDF is larger than or equal to what was found in X and F
        under the null hypothesis that X was sampled from F.

    """
    
    N = X.shape[0]
    ecdf = empirical_cdf(X)
    D = np.sqrt(N)*(ecdf.squeeze() - F.squeeze()).abs().max().item()
    # Vrbik correction
    if correction:
        D = D + 1/(6*np.sqrt(N))+(D-1)/(4*N)
    return 1 - kolmogorov_cdf(D)


def trapezoidal_rule(x, y, dim = 0, keepdim = False):
    """Trapezoidal integration.
    

    Parameters
    ----------
    x : torch.Tensor of any shape
        Tensor of the independent variable.
    y : torch.Tensor of the same shape as x
        Tensor of the dependent variable. Should be the same shape as x.
    dim : int, optional
        The dimension to integrate over. The default is 0.
    keepdim : bool, optional
        If keepdim is True, the output tensor is of the same size as y except
        along dimension dim which will be of size 1. Otherwise this dimension
        is squeezed.

    Returns
    -------
    integral : torch.Tensor
        The integral along dimension dim. The shape is the same as y except
        with dimension dim removed or set to size 1 (see keepdim argument).

    """
    n = x.shape[dim]
    ind = torch.arange(n).to(device)
    
    h = (y.index_select(dim,ind[1:])+y.index_select(dim,ind[:-1]))/2
    integral = (h*x.diff(dim = dim)).sum(dim, keepdim = keepdim)
    
    return integral


# END:   Loss and other functions
# =============================================================================

