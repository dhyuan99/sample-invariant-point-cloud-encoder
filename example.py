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
    # plt.plot(test_input.squeeze().detach().cpu(), test_target.detach().cpu(), color='black', linewidth=3, label='GT Function')
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

def reconstruction_2d(f, FE, savepath):
    """
    Visualize 2d function reconstruction.
    
    Args:
        f (Function object): a random 2d function.
        FE (FPE_Encoder or BVE_Encoder): the function encoder.
        savepath (str): where to store the visualization.

    Output:
        An image saved at the savepath.
        Reconstruction error and corrcoef is printed.
    """ 
    train_input, train_target = f.random_sample(10000)
    train_input, train_target = train_input[None,...], train_target[None,...]
    Vf = FE.encode(train_input, train_target)

    x = torch.linspace(0, 1, 64, device=FE.device)
    xx, yy = torch.meshgrid(x, x, indexing='ij')
    test_input = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    test_target = f.predict(test_input).reshape(64, 64)
    test_pred = FE.query(Vf, test_input).reshape(64, 64)
    test_pred = test_pred - test_pred.mean() + test_target.mean()

    recon_err = (test_target-test_pred).abs().mean().item()
    print(f'2d reconstruction error {savepath.strip(".jpg")}: {round(recon_err, 3)}.')
    corrcoef = torch.corrcoef(
        torch.stack([test_target.reshape(-1), test_pred.reshape(-1)], dim=0)
    )[0,1].item()
    print(f'2d corrcoef {savepath.strip(".jpg")}: {round(corrcoef, 3)}.')

    test_target = test_target.detach().cpu()
    test_pred = test_pred.detach().cpu()
    ax1 = plt.subplot(1,2,1)
    ax1.pcolor(test_target, vmin=0, vmax=1, cmap='bwr')
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title('ground-truth')
    ax2 = plt.subplot(1,2,2)
    ax2.pcolor(test_pred, vmin=0, vmax=1, cmap='bwr')
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title('reconstruction')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = Function(1, 5, device=device, seed=3)

    f.random_sample(1000)

    xx = torch.rand(size=(1000, 1), device=device)
    xx = 1 - xx ** 2
    yy = f.predict(xx)
    yy = yy + torch.randn_like(yy) * 0.1
    dist = (xx-xx.T)**2 + (yy.reshape(-1,1)-yy.reshape(1,-1))**2
    dist.fill_diagonal_(999)
    intensity = 1-100*torch.min(dist, axis=0).values
    intensity = intensity ** 80
    xx, yy, intensity = xx.cpu(), yy.cpu(), intensity.cpu()
    plt.hist(intensity)
    plt.savefig('df.jpg')
    plt.close()

    plt.figure(figsize=(4,4))
    plt.scatter(xx, yy, s=1, c=intensity, cmap='bwr')
    x = torch.linspace(0,1,100, device=device).reshape(100,1)
    y = f.predict(x)
    x, y = x.cpu(), y.cpu()
    plt.plot(x, y, linewidth=3, color='black')
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig('hi.jpg')
    plt.close()


    xx = torch.rand(size=(1000, 1), device=device)
    xx = xx ** 2
    yy = f.predict(xx)
    yy = yy + torch.randn_like(yy) * 0.1

    dist = (xx-xx.T)**2 + (yy.reshape(-1,1)-yy.reshape(1,-1))**2
    dist.fill_diagonal_(999)
    intensity = 1-100*torch.min(dist, axis=0).values
    intensity = intensity ** 80

    xx, yy, intensity = xx.cpu(), yy.cpu(), intensity.cpu()

    plt.figure(figsize=(4,4))
    plt.scatter(xx, yy, s=1, c=intensity, cmap='bwr')
    plt.plot(x, y, linewidth=3, color='black')
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig('hi2.jpg')
    plt.close()


    f = Function(1, 5, device=device)
    FE = HDFE.FPE_Encoder(1, 16000, 40, device, seed=2)
    reconstruction_1d(f, FE, 'examples/recon_FPE_high_50.jpg')

    FE = HDFE.FPE_Encoder(1, 16000, 20, device, seed=1)
    reconstruction_1d(f, FE, 'examples/recon_FPE_low_50.jpg')

    FE = HDFE.FPE_Encoder(1, 1000, 10, device, seed=0)
    reconstruction_1d(f, FE, 'examples/recon_FPE_10.jpg')

    FE = HDFE.BVE_Encoder(1, 16000, 2000, device)
    reconstruction_1d(f, FE, 'examples/recon_BVE_1d.jpg')
    
    f = Function(2, 5, device=device)
    FE = HDFE.FPE_Encoder(2, 16000, 27, device)
    HDFE.get_receptive_field(FE, 'examples/recep_FPE.jpg')
    reconstruction_2d(f, FE, 'examples/recon_FPE_2d.jpg')
    
    FE = HDFE.BVE_Encoder(2, 16000, 2000, device)
    HDFE.get_receptive_field(FE, 'examples/recep_BVE.jpg')
    reconstruction_2d(f, FE, 'examples/recon_BVE_2d.jpg')


