import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
import HDFE

class Function:
    def __init__(self, input_dim, num_coefs, device, seed):
        """ Random function generator, mapping from (0,1)^n -> (0,1)

        Args:
            input_dim (int): input dimension n.
            num_coefs (int): determines the complexity of the function.
            device (torch.device): cpu or gpu.
        """

        self.input_dim = input_dim
        self.num_coefs = num_coefs
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)
        self.M = torch.randn((input_dim,), device=device)
        self.coef = torch.randn((num_coefs,), device=device)

        self.my, self.My = None, None

    def random_sample(self, num_samples):
        """ Sample the function with uniform distribution in (0,1).

        Args:
            num_samples (int): number of samples being drawn.

        Returns:
            x: (torch.tensor): has shape (num_samples, input_dim)
            y: (torch.tensor): has shape (num_samples, )
        """
        x = torch.rand(
            size=[num_samples, self.input_dim], 
            device=self.device
        )
        tx = x @ self.M
        tx = torch.stack(
            [torch.sin(2*3.14159*k*tx) for k in range(self.num_coefs)], dim=-1
        )
        y = tx @ self.coef
        self.my, self.My = y.min(), y.max()
        y = (y-self.my) / (self.My-self.my)
        return x, y

    def predict(self, x):
        """ Predict the function value with input x

        Args:
            x (torch.tensor): has shape (num_samples, input_dim)

        Returns:
            yhat (torch.tensor): has shape (num_samples)
        """
        tx = x @ self.M
        tx = torch.stack(
            [torch.sin(2*3.14159*k*tx) for k in range(self.num_coefs)], dim=-1
        )
        y = tx @ self.coef
        y = (y-self.my) / (self.My-self.my)
        return y

def reconstruction_1d(f, FE, savepath):
    """
    Visualize 1d function reconstruction.
    
    Args:
        f (Function object): a random 1d function.
        FE (FPE_Encoder or BVE_Encoder): the function encoder.
        savepath (str): where to store the visualization.

    Output:
        An image saved at the savepath.
        Reconstruction error is printed.
    """ 
    train_input, train_target = f.random_sample(10000)
    Vf = FE.encode(train_input, train_target)

    test_input = torch.linspace(0,1,100, device=FE.device).reshape(-1,1)
    test_target = f.predict(test_input).squeeze()
    test_pred = FE.query(Vf.squeeze(), test_input)
    test_pred = test_pred - test_pred.mean() + test_target.mean()
    test_input = test_input.squeeze()
    plt.figure(figsize=(4,4))
    plt.plot(test_input.squeeze().detach().cpu(), test_target.detach().cpu(), color='black', linewidth=3, label='GT Function')
    plt.plot(test_input.squeeze().detach().cpu(), test_pred.detach().cpu(), color='red', linewidth=3, label='Fitted Function')
    plt.xticks([]); plt.yticks([])
    # plt.legend(fontsize=15, loc='upper right')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

    recon_err = torch.mean(torch.abs(test_target-test_pred)).item()
    print(f'1d reconstruction error {savepath.strip(".jpg")}: {round(recon_err, 3)}.')
    corrcoef = torch.corrcoef(
        torch.stack([test_target.reshape(-1), test_pred.reshape(-1)], dim=0)
    )[0,1].item()
    print(f'1d corrcoef {savepath.strip(".jpg")}: {round(corrcoef, 3)}.')

def isometry(FE, savepath):
    pairs = []
    for _ in range(1000):
        f1 = Function(1, 5, device=device, seed=None)
        x1, y1 = f1.random_sample(1000)
        f2 = Function(1, 5, device=device, seed=None)
        x2, y2 = f2.random_sample(1000)

        Vf1 = FE.encode(x1, y1)
        Vf2 = FE.encode(x2, y2)
        vector_similarity = HDFE.FPE_Encoder.similarity(Vf1[None,:], Vf2[None,:])

        x = torch.linspace(0, 1, 100).reshape(-1, 1).to(device)
        y1 = f1.predict(x)
        y2 = f2.predict(x)
        function_distance = (y1-y2).abs().mean()

        pairs.append([function_distance.item(), vector_similarity.item()])

    dists = [dist for dist, _ in pairs]
    sims = [sim for _, sim in pairs]

    plt.scatter(dists, sims)
    plt.xlabel("Function Distance")
    plt.ylabel("Function Vector Similarity")
    plt.savefig('examples/isometry.jpg')
    plt.close()
    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate a 1d function.
    f = Function(1, 5, device=device, seed=0)

    #####################################################################
    #####################################################################
    ############ Fractional Power Encoding Reconstruction ###############
    #####################################################################
    #####################################################################
    
    # Function Encoder 1: good dim, good alpha.
    FE1 = HDFE.FPE_Encoder(input_dim=1, dim=8000, alpha=30, device=device, seed=1)
    reconstruction_1d(f, FE1, 'examples/recon_FPE_good_alpha.jpg')

    # Function Encoder 2: good dim, too small alpha
    FE2 = HDFE.FPE_Encoder(input_dim=1, dim=8000, alpha=10, device=device, seed=1)
    reconstruction_1d(f, FE2, 'examples/recon_FPE_low_alpha.jpg')

    # Function Encoder 3: small dim, good alpha 
    FE3 = HDFE.FPE_Encoder(input_dim=1, dim=1000, alpha=30, device=device, seed=1)
    reconstruction_1d(f, FE3, 'examples/recon_FPE_low_dim.jpg')

    #####################################################################
    #####################################################################
    ############# Binary Vector Encoding Reconstruction #################
    #####################################################################
    #####################################################################
    FE = HDFE.BVE_Encoder(input_dim=1, dim=8000, q=2000, device=device)
    reconstruction_1d(f, FE, 'examples/recon_BVE_1d.jpg')
    
    # Isometry property.
    FE = HDFE.FPE_Encoder(input_dim=1, dim=4000, alpha=25, device=device, seed=0)
    isometry(FE, 'examples/isometry.jpg')





