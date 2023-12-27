import torch.nn as nn
import torch


class Dice(nn.Module):
    def __init__(self, num_features, dim=2):
        super(Dice, self).__init__()
        self.alphas = nn.Parameter(torch.zeros(num_features))


    def forward(self, x, axis=-1, epsilon=1e-9):
        input_shape = list(x.size())
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]

        # Case: train mode (uses stats of the current batch)
        mean = torch.mean(x, dim=reduction_axes)
        broadcast_mean = mean.view(broadcast_shape)
        std = torch.mean(torch.square(x - broadcast_mean) + epsilon, dim=reduction_axes)
        std = torch.sqrt(std)
        broadcast_std = std.view(broadcast_shape)
        x_normed = (x - broadcast_mean) / (broadcast_std + epsilon)
        x_p = torch.sigmoid(x_normed)

        return self.alphas * (1.0 - x_p) * x + x_p * x
        

if __name__ == "__main__":
    # Example usage:
    x = torch.randn((10, 20, 30))
    dice = Dice(x.shape[-1])
    y = dice(x)
    print(y.shape)