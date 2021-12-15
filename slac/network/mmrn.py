import torch
import torch.nn as nn
from slac.network.Keypoint_Encoder import KeypointNet
from slac.network.Keypoint_Encoder import contrastive_alignment_loss

class MMRN(nn.Module):
    """
    Multi-Modal Representation Network (MMRN) abstracts fusing of visual
    and tactile data.
    """
    def __init__(self):
        super().__init__()
        self.visual_encoder = KeypointNet()
        self.tactile_encoder = MLP(input_size=6, output_size=64, hidden_sizes=[64])
        self.action_encoder = MLP(input_size=3, output_size=64, hidden_sizes=[64])

    def encode_visual(self, rgbd, projection_inv, view_inv):
        z_v = self.visual_encoder(rgbd, projection_inv, view_inv)
        return z_v

    def encode_tactile(self, tactile):
        z_t = self.tactile_encoder(tactile)
        return z_t

    def encode_action(self, action):
        z_a = self.action_encoder(action)
        return z_a

    def align_contrastive(self, true_align, negative_align):
        return contrastive_alignment_loss(true_align, negative_align)

    def forward(self, rgbd, projection_inv, view_inv, tactile, action, fuse_latent=True):
        """
        Return combined visual and tactile representation.
        """
        B, S, C, H, W = rgbd.size()
        rgbd = rgbd.view(B * S, C, H, W)
        action = action.view(B * S, -1)
        tactile = tactile.view(B * S, -1)
        z_v = self.encode_visual(rgbd, projection_inv, view_inv)
        z_t = self.encode_tactile(tactile)
        z_a = self.encode_action(action)
        if fuse_latent:
            z = torch.cat([z_v, z_t, z_a], dim=1)
            z = z.view(B, S, -1)
            return z
        else:
            z_v = z_v.view(B, S, -1)
            z_t = z_t.view(B, S, -1)
            z_a = z_a.view(B, S, -1)
            return z_v, z_t, z_a

    def loss_forward(self, rgbd, projection_inv, view_inv, tactile, action, fuse_latent=True, **kwargs):
        B, S, C, H, W = rgbd.size()
        rgbd = rgbd.view(B * S, C, H, W)
        action = action.view(B * S, -1)
        tactile = tactile.view(B * S, -1)
        z_v, sep_loss, s_loss = self.visual_encoder.loss_forward(rgbd, projection_inv, view_inv, **kwargs)
        # If needed, these encoders can also provide losses here.
        z_t = self.encode_tactile(tactile)
        z_a = self.encode_action(action)

        if fuse_latent:
            z = torch.cat([z_v, z_t, z_a], dim=1)
            z = z.view(B, S, -1)
            return z, sep_loss, s_loss
        else:
            z_v = z_v.view(B, S, -1)
            z_t = z_t.view(B, S, -1)
            z_a = z_a.view(B, S, -1)
            return (z_v, z_t, z_a), sep_loss, s_loss

class MLP(nn.Module):
    """
    Model + code based on: https://github.com/wilson1yan/contrastive-forward-model/blob/master/cfm/models.py
    """

    def __init__(self, input_size=64, output_size=128, hidden_sizes=[256, 256]):
        """
        Args:
        - input_size (int): input dim
        - output_size (int): output dim
        - hidden_sizes (list): list of ints with hidden dim sizes
        """
        super().__init__()
        # TODO: Parameterize activation function?

        if hidden_sizes is None:
            hidden_sizes = []
        model = []
        prev_h = input_size
        for h in hidden_sizes + [output_size]:
            model.append(nn.Linear(prev_h, h))
            model.append(nn.ReLU())
            prev_h = h
        model.pop()  # Pop last ReLU
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


