# %%
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
import time
# %%
class Mnist(torch.nn.Module):
    def __init__(self, dims: tuple[int, ...], device: torch.device, dtype: torch.dtype):
        super(Mnist, self).__init__()

        self.n = len(dims) - 1
        for i in range(self.n):
            setattr(
                self,
                "ln_{}".format(i),
                torch.nn.Linear(dims[i], dims[i+1], device=device, dtype=dtype)
            )
    
    def forward(self, x: torch.Tensor):
        for i in range(self.n):
            ln = getattr(self, "ln_{}".format(i))
            x = ln.forward(x)
            if (i+1) != self.n:
                x = x.relu()
        return x

def main():
    dir_data = "/root/tasks/wnum_dev/data/torch"
    batch_size = 256
    epoch = 32
    lr = 0.01
    dims = (784, 128, 10)
    device = torch.device('cpu')
    dtype = torch.float32

    m = Mnist(dims=dims, device=device, dtype=dtype)
    data = torch.utils.data.DataLoader(
        datasets.MNIST(
            root=dir_data,
            train=True,
            transform=transforms.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True
    )
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(
        m.parameters(),
        lr=lr
    )
    for epoch_idx in range(epoch):
        loss_all = 0.0
        bar = tqdm(data, desc="{}/{}".format(epoch_idx+1, epoch))
        for data_inp, data_opt in bar:
            data_inp = data_inp.flatten(1).to(device=device, dtype=dtype)
            data_opt = torch.nn.functional.one_hot(
                data_opt,
                num_classes=10
            ).to(device=device, dtype=dtype)
            opt.zero_grad()
            pred = m(data_inp)
            loss_data = loss.forward(pred, data_opt)
            loss_all += float(loss_data.abs().detach().numpy())
            bar.set_description("Epoch {}/{}, Loss: {:.4f}".format(epoch_idx+1, epoch, loss_all))
            loss_data.backward()
            opt.step()


if __name__ == "__main__":
    main()