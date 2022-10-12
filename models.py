from typing import List
import torch
from escnn import nn
import escnn


class EqvModel(nn.EquivariantModule):
    def __init__(self, hidden_dims=[6, 6, None, 12, 12, None, 24, 24, None, 48, 48, None, 96, 96, None, 96, 96]):
        super().__init__()
        n_rotations = 4
        rot_2d = escnn.gspaces.flipRot2dOnR2(N=n_rotations)

        self.input_type = escnn.nn.FieldType(rot_2d, 3 * [rot_2d.trivial_repr])
        self.output_type = escnn.nn.FieldType(rot_2d, hidden_dims[0] * [rot_2d.regular_repr])

        self.conv = escnn.nn.R2Conv(self.input_type, self.output_type, 7, 3)

        layers: List[nn.EquivariantModule] = [self.conv, nn.ReLU(self.output_type)]

        in_type = self.output_type
        for dim in hidden_dims:
            if dim is None:
                layers.append(nn.PointwiseAvgPool(in_type, 2, 2))
            else:
                out_type = nn.FieldType(rot_2d, dim * [rot_2d.regular_repr])
                layers.append(nn.R2Conv(in_type, out_type, kernel_size=7, padding=3))
                in_type = out_type
                layers.append(nn.ReLU(in_type))

        layers.append(nn.PointwiseAdaptiveMaxPool(in_type, 1))

        self.linear_layer = torch.nn.Linear(hidden_dims[-1] * n_rotations * 2, 2)



        self.hidden_dims = hidden_dims
        self.layers = layers
        self.model = nn.SequentialModule(*layers)

        # self.layers = nn.ModuleList([
        # ])
    
    def forward(self, x):
        y = self.model(self.input_type(x))
        y = y.tensor
        y = y.view(y.shape[0], y.shape[1])
        # print(f"{y.tensor.shape=}")
        y = self.linear_layer(y)
        return y

    def evaluate_output_shape(self, x):
        b, c, w, h = x.shape
        scale = 2 ** sum((1 for d in self.hidden_dims if d is None))

        return (b, c, w // scale, h // scale)

if __name__ == "__main__":
    m = EqvModel()
    x = torch.randn(2, 3, 96, 96)
    y = m(x)

    sd = m.state_dict()
    torch.save(sd, 'eqv_model.pt')
    m.load_state_dict(torch.load('eqv_model.pt'))
    y = m(x)

    print(y)
    print(f"{y.shape=}")
