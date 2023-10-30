"""
Author: Dehao Yuan; Email: dhyuan@umd.edu

This file contains all the necessity for running HDFE.
It provides two implementations, discrete HDFE (BVE_Encoder) and continuous HDFE (FPE_Encoder).
BVE stands for binary vector encoding; FPE stands for fractional power encoding.
    * BVE is faster, computationally efficient, memory efficient, but less accurate.
    * FPE is slower, memory intense, but more accurate.
The most important functionalities in this file is `BVE_Encoder` and `FPE_Encoder`.
They share the three member function prototypes (most important functionalities):
    * __init__(self, input_dim, dim, ...): constructor of the function encoder.
    * encode(self, x, y): produce function vector given function samples y = f(x).
    * query(self, Vf, x): query the function vector Vf with inputs x.
There is a useful tool to inspect the property of your function encoder `get_receptive_field`.
It instructs how to set the parameter of the function encoder.

Good luck hacking!
"""

import torch
from sklearn.svm import OneClassSVM

def get_array_size(arr):
    """
    Return the total memory usage of an array in Mb.
    """
    return arr.element_size() * arr.nelement() / 1e6

def get_receptive_field(FE, savepath, reso=64):
    import matplotlib.pyplot as plt
    """
    Get the statistics of a function encoder.

    Args:
        FE (FPE_Encoder or BVE_Encoder): function encoder
        savepath (str): path to save the visualization.
        reso (int, optional): resolution of visualization. Defaults to 64.

    Output:
        figure showing the similarity change. receptive field is labeled.
    """
    input_dim = FE.input_dim
    x = torch.linspace(0, 1, reso, device=FE.device)
    xx = torch.stack([x]*input_dim, dim=-1)
    Vx = FE.encode_x(xx)
    sim_x = FE.similarity(Vx[[0]], Vx).squeeze()
    recep_field = torch.argmin(torch.clamp(sim_x, 0.01, 1)) / reso

    y = torch.linspace(0, 1, reso, device=FE.device)
    Vy = FE.encode_y(y)
    sim_y = FE.similarity(Vy[[0]], Vy).squeeze()

    x, y, sim_x, sim_y, recep_field = x.cpu(), y.cpu(), sim_x.cpu(), sim_y.cpu(), recep_field.cpu()
    plt.figure(figsize=(8,4))
    ax1 = plt.subplot(1,2,1)
    ax1.annotate(f'{recep_field}', (recep_field, 0.01))
    ax1.scatter(recep_field, 0.01)
    ax1.plot(x, sim_x)
    ax1.title.set_text('Similarity of X')
    ax2 = plt.subplot(1,2,2)
    ax2.plot(y, sim_y) 
    ax2.title.set_text('Similarity of Y')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


class BVE_Encoder:
    def __init__(self, input_dim, dim, q, device, alpha=0.99): 
        """
        * The input vector should be normalized so that the element ranges between 0 and 1.
            The (0,1) interval will be quantized into `q` sub-intervals. 
            All numbers within a sub-interval will be represented as the same binary vector.

        Args:
            input_dim (int): the dimension of the input real vector.
            dim (int): the dimension of the output binary vector.
            q (int): quantization resolution.
            device (torch.device): cpu or cuda.
            alpha (float): determines the receptive field of HDFE.
        """
        self.input_dim, self.dim, self.q = input_dim, dim, q
        self.device = device
        self.Ex = self.BinaryXEncoder(input_dim, dim, q, device, alpha)
        self.Ey = self.BinaryYEncoder(dim, q, device)

    def encode_x(self, x):
        """
        Encode samples input into a binary vector.

        Args:
            x (torch.Tensor): has shape (num_samples, input_dim)

        Return:
            Vx (torch.Tensor bool): has shape (num_samples, dim)
        """
        return self.Ex.encode(x) 

    def encode_y(self, y):
        """
        Encode samples output into a binary vector.

        Args:
            y (torch.Tensor): has shape (num_samples, )

        Return:
            Vy (torch.Tensor bool): has shape (num_samples, dim)
        """
        return self.Ey.encode(y)

    def encode(self, x, y):
        """
        Encode samples of functions into a binary vector.

        Args:
            x (torch.Tensor): has shape (num_samples, input_dim)
            y (torch.Tensor): has shape (num_samples,)

        Return:
            Vf (torch.Tensor bool): has shape (dim)
        """
        Vx = self.Ex.encode(x)
        Vy = self.Ey.encode(y)
        Vf = torch.mean(torch.logical_xor(Vx, Vy).float(), dim=0) > 0.5
        Vf = Vf.squeeze()
        return Vf

    def query(self, Vf, x):
        """
        Query the function vector with input x.

        Args:
            Vf (torch.Tensor bool): has shape (dim, )
            x (torch.Tensor): has shape (batch_size, input_dim)

        Return:
            yhat (torch.Tensor): has shape (batch_size,)

        """
        Vx = self.Ex.encode(x)
        Vf = Vf.reshape(1, self.dim)
        Vy_hat = torch.logical_xor(Vx, Vf)
        return self.Ey.decode(Vy_hat)
    
    @staticmethod
    def randbinary(size, device, p=0.5):
        """
        Generate random bipolar vector {-1,1} of size = `size`.
        An element is 1 with prob. = `p`.

        Args:
            size (tuple): size of the array
            device (torch.device): cuda or cpu.
            p (float, optional): prob. (element = 1). Defaults to 0.5.
        """
        a = torch.rand(size=size, device=device) < p
        a = 2*a - 1
        return a

    @staticmethod
    def similarity(a, b):
        """
        Compute the similarity between two binary vectors.
        
        Args:
            a: has shape (p, n)
            b: has shape (q, n)

        Return:
            res: similarity of shape (p, q)
        """ 
        tmp_a = 2*a - 1
        tmp_b = 2*b - 1
        return tmp_a.float() @ tmp_b.float().T / tmp_a.shape[1]

    @staticmethod
    class BinaryXEncoder:
        def __init__(self, input_dim, dim, q, device, alpha=0.99, verbose=False):
            """
            BinaryXEncoder will encode a real vector of length `input_dim` into a binary vector of length `dim`.
            It aims for property:
                * similarity(E(x), E(x+dx)) close to 1 when |dx| < eps_0
                * similarity(E(x), E(x+dx)) close to 0 when |dx| > eps_0.

            * The input vector should be normalized so that the element ranges between 0 and 1.
                The (0,1) interval will be quantized into `q` sub-intervals. 
                All numbers within a sub-interval will be represented as the same binary vector.

            Args:
                input_dim (int): the dimension of the input real vector.
                dim (int): the dimension of the output binary vector.
                q (int): quantization resolution.
                alpha (float): determines the receptive field of HDFE.
                verbose (bool): whether printing logging information.
            """

            self.input_dim, self.dim, self.q, self.alpha = input_dim, dim, q, alpha
            self.device, self.verbose = device, verbose

            def gen_level_vecs(input_dim, q, dim, alpha):
                V = torch.zeros([input_dim, q, dim], device=device)
                for i in range(input_dim):
                    V[i,0] = BVE_Encoder.randbinary(size=(dim,), device=device, p=0.5)
                    for j in range(1, q):
                        V[i,j] = V[i,j-1] * BVE_Encoder.randbinary(size=(dim,), device=device, p=alpha)
                V = (V > 0)
                return V

            self.V = gen_level_vecs(input_dim, q, dim, alpha)
            self.V = self.V.to(device)

            if verbose:
                print(f'The dictionary has shape {self.V.shape};')
                print(f'The dictionary has size {get_array_size(self.V)} Mb.')
                print(f'The receptive field is {self.get_receptive_field()}.')
                print()

        def encode(self, x):
            """ Encode a real vector `x` into a binary vector.

            Args:
                x (torch.Tensor): has shape (batch_size, input_dim)
            
            Return:
                res (torch.Tensor bool): has shape (batch_size, dim)
            """

            batch_size, input_dim = x.shape
            x = x.reshape(-1)
            qx = (x * (self.q-1) + 0.5).type(torch.long)
            ind = torch.arange(input_dim, device=x.device).repeat(batch_size)
            res = self.V[ind, qx].reshape(batch_size, input_dim, self.dim)

            if self.verbose:
                print(f'processing #points: {batch_size}.')
                print(f'memory usage: {get_array_size(res)} Mb.')
                print()

            num_falses = self.input_dim - torch.sum(res, dim=1)
            res = num_falses % 2 == 0
            return res

        def get_receptive_field(self):
            sim = self.similarity(self.V[0,[self.q//2]], self.V[0])
            eps_0 = torch.sum(sim > 0.5) /  self.q
            return round(eps_0.item(), 3)

    @staticmethod
    class BinaryYEncoder:
        def __init__(self, dim, q, device):
            """
            BinaryYEncoder encodes a real number in (0,1) into a binary vector.
            It aims for the property
                * similarity(E(x), E(y)) > 0 for all x, y.
                * similarity(E(0), E(y)) = 0.

            Args:
                dim (int): dimension of the output vector.
                q (int): quantization resolution
                device (torch.device): cuda or cpu.
            """

            self.dim, self.q, self.device = dim, q, device
            start = torch.rand(size=(dim,), device=device)
            end = torch.rand(size=(dim,), device=device)
            self.V = torch.zeros([q, dim], device=device)
            for i, k in enumerate(torch.linspace(0, 1, steps=q)):
                self.V[i] = k*start + (1-k)*end
            self.V = self.V > 0.5 

        def encode(self, y):
            """
            Encode a real number into a binary vector.
            
            Args:
                y (torch.Tensor): has shape (batch_size, )

            return:
                out (torch.Tensor bool): has shape (batch_size, dim)
            """ 

            qy = (y * (self.q-1) + 0.5).type(torch.long)
            return self.V[qy]

        def decode(self, Vy):
            """
            Decode a binary vector and recover the y-value.

            Args:
                Vy (torch.Tensor bool): has shape (batch_size, dim)

            Return:
                out (torch.Tensor): has shape (batch_size)
            """
            sim = BVE_Encoder.similarity(Vy, self.V)
            yhat = torch.argmax(sim, dim=1)
            yhat = (yhat.type(torch.float64)) / (self.q-1)
            return yhat


class FPE_Encoder:
    def __init__(self, input_dim, dim, alpha, device, seed):
        torch.manual_seed(seed)
        self.Ex = alpha * torch.randn((input_dim, dim), device=device)
        self.Ey = torch.randn((dim,), device=device)
        self.T = self.Ey[:,None] - self.Ey[None,:]
        self.input_dim = input_dim
        self.dim = dim
        self.alpha = alpha
        self.device = device

    def encode_x(self, x):
        """
        Encode the function input into a vector.

        Args:
            x (torch.Tensor): function input samples. (num_samples, input_dim)
        
        Return:
            Vx (torch.Tensor): (num_samples, dim)
        """
        return torch.exp(1j * (x @ self.Ex))

    def encode_y(self, y):
        """
        Encode the function output into a vector.

        Args:
            y (torch.Tensor): function output samples. (num_samples, )

        Return:
            Vy (torch.Tensor): (num_samples, dim)
        """
        return torch.exp(1j * (torch.outer(y, self.Ey)))

    def encode(self, x, y):
        """ Encode function samples into a vector

        Args:
            x (torch.Tensor): function input samples. (num_samples, input_dim)
            y (torch.Tensor): function output samples. (num_samples)

        Return:
            Vf (torch.Tensor, complex64): function vector. (dim)
        """

        assert len(x.shape) == 2 and len(y.shape) == 1, \
            "x should has shape (num_samples, input_dim), y should have shape (num_samples)."
        assert y.shape[0] == x.shape[0], \
            "x and y must have the same number of samples."

        with torch.no_grad():
            zx = torch.exp(1j * (x @ self.Ex))
            zy = torch.exp(1j * (torch.outer(y, self.Ey)))
        return torch.mean(zx*zy, dim=0).squeeze()

    def robust_encode(self, x, y):
        zx = torch.exp(1j * (x @ self.Ex))
        zy = torch.exp(1j * (torch.outer(y, self.Ey)))
        zxy = zx*zy
        K = torch.absolute(zxy @ torch.conj(zxy).T)
        model = OneClassSVM(kernel='precomputed').fit(K)
        out = torch.tensor(model.dual_coef_, dtype=torch.complex64) @ zxy[model.support_]
        out = out.to(self.device)
        out = out / torch.norm(out, dim=1, keepdim=True)
        return out

    def optim_target(self, z):
        """ Decoding objective function
        return argmax_y <exp(1j * y@Ey), z>

        Args:
            z (torch.Tensor complex64): (batch_size, dim)

        Returns:
            yhat (torch.Tensor float32): (batch_size, )
        """

        num_samples, dim = z.shape

        a = z.abs()
        w = z.angle()

        y = torch.zeros((num_samples), device=self.device) + 0.5
        for _ in range(501):
            r = torch.randperm(dim)[:1000]
            A = a[:,None,r] * a[:,r,None]
            T = self.Ey[None,None,r] - self.Ey[None,r,None]
            W = w[:,None,r] - w[:,r,None]

            grad = torch.mean(A * torch.sin(T*y[:,None,None] - W) * T, dim=[1,2])
            y = y - grad

        return y

    def query(self, Vf, x):
        """ Query the function vector Vf with query points x.

        Args:
            Vf (torch.Tensor): function vector. (dim, )
            x (torch.Tensor): query points. (batch_size, input_dim)

        Return:
            yhat (torch.Tensor): has shape (batch_size, )
        """
        zx = torch.exp(1j * (x @ self.Ex))
        zy = Vf / zx

        yhat = self.optim_target(zy)
        return yhat

    @staticmethod
    def similarity(a, b):
        """
        Compute the similarity between two complex vectors.
        
        Args:
            a: has shape (p, n)
            b: has shape (q, n)

        Return:
            res: similarity of shape (p, q)
        """ 
        a_norm = torch.sqrt(torch.sum((a * torch.conj(a)), dim=1, keepdim=True))
        b_norm = torch.sqrt(torch.sum((b * torch.conj(b)), dim=1, keepdim=True)).T
        return torch.absolute(a @ torch.conj(b).T) / torch.absolute(a_norm*b_norm)

