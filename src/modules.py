import math
import warnings
import torch
import torch.nn as nn
 
# implement setcnn https://github.com/YannDubs/Neural-Process-Family/blob/master/npf/architectures/setcnn.py

def init_kernels(n_heads, n_kernel, slope=10., noise=.1):
    step = 1./n_kernel
    kernels = torch.exp(-slope*torch.arange(0, 1, step=step)).unsqueeze(0).repeat(n_heads, 1) + torch.randn(n_heads, n_kernel) * noise
    return kernels.div(kernels.sum(dim=[-1], keepdim=True)).unsqueeze(1).unsqueeze(1)
 
def weights_init(module, **kwargs):
    """Initialize a module and all its descendents.
    Parameters
    ----------
    module : nn.Module
       module to initialize.
    """
    module.is_resetted = True
    for m in module.modules():
        try:
            if hasattr(module, "reset_parameters") and module.is_resetted:
                # don't reset if resetted already (might want special)
                continue
        except AttributeError:
            pass
 
        if isinstance(m, torch.nn.modules.conv._ConvNd):
            # used in https://github.com/brain-research/realistic-ssl-evaluation/
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", **kwargs)
        elif isinstance(m, nn.Linear):
            linear_init(m, **kwargs)
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
 
def get_activation_name(activation):
    """Given a string or a `torch.nn.modules.activation` return the name of the activation."""
    if isinstance(activation, str):
        return activation
 
    mapper = {
        torch.nn.LeakyReLU: "leaky_relu",
        torch.nn.ReLU: "relu",
        torch.nn.Tanh: "tanh",
        torch.nn.Sigmoid: "sigmoid",
        torch.nn.Softmax: "sigmoid",
    }
    for k, v in mapper.items():
        if isinstance(activation, k):
            return k
 
    raise ValueError("Unkown given activation type : {}".format(activation))
 
def linear_init(module, activation="leaky_relu"):
    """Initialize a linear layer.
    Parameters
    ----------
    module : nn.Module
       module to initialize.
    activation : `torch.nn.modules.activation` or str, optional
        Activation that will be used on the `module`.
    """
    x = module.weight
 
    if module.bias is not None:
        module.bias.data.zero_()
 
    if activation is None:
        return torch.nn.init.xavier_uniform_(x)
 
    activation_name = get_activation_name(activation)
 
    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return torch.nn.init.kaiming_uniform_(x, a=a, nonlinearity="leaky_relu")
    elif activation_name == "relu":
        return torch.nn.init.kaiming_uniform_(x, nonlinearity="relu")
    elif activation_name in ["sigmoid", "tanh"]:
        return torch.nn.init.xavier_uniform_(x, gain=get_gain(activation))
 
def get_gain(activation):
    """Given an object of `torch.nn.modules.activation` or an activation name
    return the correct gain."""
    if activation is None:
        return 1
 
    activation_name = get_activation_name(activation)
 
    param = None if activation_name != "leaky_relu" else activation.negative_slope
    gain = torch.nn.init.calculate_gain(activation_name, param)
 
    return gain
 
  
class MLP(torch.nn.Module):
    """General MLP class.
    Parameters
    ----------
    input_size: int
    output_size: int
    hidden_size: int, optional
        Number of hidden neurones.
    n_hidden_layers: int, optional
        Number of hidden layers.
    activation: callable, optional
        Activation function. E.g. `nn.RelU()`.
    is_bias: bool, optional
        Whether to use biaises in the hidden layers.
    dropout: float, optional
        Dropout rate.
    is_force_hid_smaller : bool, optional
        Whether to force the hidden dimensions to be smaller or equal than in and out.
        If not, it forces the hidden dimension to be larger or equal than in or out.
    is_res : bool, optional
        Whether to use residual connections.
    """
 
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=32,
        n_hidden_layers=1,
        activation=torch.nn.LeakyReLU(),
        is_bias=True,
        dropout=0,
        is_force_hid_smaller=False,
        is_res=False,
    ):
        super().__init__()
 
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.is_res = is_res
 
        if is_force_hid_smaller and self.hidden_size > max(
            self.output_size, self.input_size
        ):
            self.hidden_size = max(self.output_size, self.input_size)
            txt = "hidden_size={} larger than output={} and input={}. Setting it to {}."
            warnings.warn(
                txt.format(hidden_size, output_size, input_size, self.hidden_size)
            )
        elif self.hidden_size < min(self.output_size, self.input_size):
            self.hidden_size = min(self.output_size, self.input_size)
            txt = (
                "hidden_size={} smaller than output={} and input={}. Setting it to {}."
            )
            warnings.warn(
                txt.format(hidden_size, output_size, input_size, self.hidden_size)
            )
 
        self.dropout = torch.nn.Dropout(p=dropout) if dropout > 0 else torch.nn.Identity()
        self.activation = activation
 
        self.to_hidden = torch.nn.Linear(self.input_size, self.hidden_size, bias=is_bias)
        self.linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.hidden_size, self.hidden_size, bias=is_bias)
                for _ in range(self.n_hidden_layers - 1)
            ]
        )
        self.out = torch.nn.Linear(self.hidden_size, self.output_size, bias=is_bias)
 
        self.reset_parameters()
 
    def forward(self, x):
        out = self.to_hidden(x)
        out = self.activation(out)
        x = self.dropout(out)
 
        for linear in self.linears:
            out = linear(x)
            out = self.activation(out)
            if self.is_res:
                out = out + x
            out = self.dropout(out)
            x = out
 
        out = self.out(x)
        return out
 
    def reset_parameters(self):
        linear_init(self.to_hidden, activation=self.activation)
        for lin in self.linears:
            linear_init(lin, activation=self.activation)
        linear_init(self.out)