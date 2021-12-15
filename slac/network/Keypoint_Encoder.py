import pdb
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from slac.utils import build_mlp
# This network design is based on https://github.com/buoyancy99/unsup-3d-keypoints.
# Code modified from https://github.com/buoyancy99/unsup-3d-keypoints/blob/main/algorithms/common/models/keypoint_net.py
# Code modified from https://github.com/buoyancy99/unsup-3d-keypoints/blob/main/algorithms/common/models/keypoint_net.py


def xy_to_uv(keypoints, image_size):
    """
    Convert from keypoints in [-1,1] to pixel locations in [0,H]/[0,W].
    """
    # Move keypoints to [0, 1]
    kp = (keypoints + 1.0) / 2.0
    kp[:, :, 0] *= image_size[1]
    kp[:, :, 1] *= image_size[0]
    return kp


def keypoints_to_image_space(keypoints, heatmap_size, image_size):
    keypoints[:, :, 0] *= image_size[1] / heatmap_size[1]
    keypoints[:, :, 1] *= image_size[0] / heatmap_size[0]
    return keypoints


def separation_loss(keypoints):
    """
    Args:
        keypoints (torch.tensor): keypoint representation BxKx3
    Returns:
        separation_loss (torch.tensor): separation loss
    """
    num_points = keypoints.shape[1]
    xy0 = keypoints[:, :, None, :].expand(-1, -1, num_points, -1)
    xy1 = keypoints[:, None, :, :].expand(-1, num_points, -1, -1)
    sq_dist = torch.sum((xy0 - xy1) ** 2, 3)
    loss = 1 / (1000 * sq_dist + 1)
    return torch.mean(loss, [1, 2])


def contrastive_alignment_loss(true_alignment, negative_alignment):
    """
    Alignment should be between [0, 1].
    """
    negative_alignment = torch.sum(negative_alignment, dim=1)

    align_loss = -torch.log(true_alignment / (true_alignment + negative_alignment))
    return align_loss


def shift_loss(model, rgbd, offset_max, vis=False, device=None):
    # Get random offset version of rgbd.
    offset_vector = ((np.random.rand(2) * 2 - 1.0) * offset_max).astype(int)
    offset_rgbd = torch.rand_like(rgbd)
    offset_rgbd[
    :, :,
    offset_vector[1] if offset_vector[1] >= 0 else 0: model.image_size[0] if offset_vector[1] >= 0 else
    model.image_size[0] + offset_vector[1],
    offset_vector[0] if offset_vector[0] >= 0 else 0: model.image_size[1] if offset_vector[0] >= 0 else
    model.image_size[1] + offset_vector[0],
    ] = rgbd[
        :, :,
        0 if offset_vector[1] >= 0 else -offset_vector[1]: model.image_size[0] - offset_vector[1] if
        offset_vector[1] >= 0 else model.image_size[0],
        0 if offset_vector[0] >= 0 else -offset_vector[0]: model.image_size[1] - offset_vector[0] if
        offset_vector[0] >= 0 else model.image_size[1],
        ]

    # Visualize shifted data for debugging purposes.
    if vis:
        orig_rgb_ax = plt.subplot(221)
        rgb = rgbd[:, :3, :, :]
        orig_rgb_ax.imshow(rgb.cpu().numpy()[0].transpose(1, 2, 0))
        orig_rgb_ax.axis('off')
        orig_depth_ax = plt.subplot(222)
        depth = rgbd[:, 3, :, :]
        orig_depth_ax.imshow(depth.cpu().numpy()[0])
        orig_depth_ax.axis('off')

        new_rgb_ax = plt.subplot(223)
        new_rgb_ax.imshow(offset_rgbd[0, :3].cpu().numpy().transpose(1, 2, 0))
        new_rgb_ax.axis('off')
        new_depth_ax = plt.subplot(224)
        new_depth_ax.imshow(offset_rgbd[0, 3].cpu().numpy())
        new_depth_ax.axis('off')
        plt.show()

    # Encode original and offset to keypoints.
    _, keypoints_original = model.encode(rgbd, noise=0.0)
    _, keypoints_shifted = model.encode(offset_rgbd, noise=0.0)

    # Scale offset vector to heatmap size.
    offset_vector[0] *= model.heatmap_size[1] / model.image_size[1]
    offset_vector[1] *= model.heatmap_size[0] / model.image_size[0]

    # Find difference after undoing shift on keypoints.
    loss = F.mse_loss(keypoints_original[:, :, :2],
                      keypoints_shifted[:, :, :2] -
                      torch.from_numpy(offset_vector).to(device)[None, None, :])
    return loss


class KeypointNet(nn.Module):
    def __init__(self, k=16, image_size=(84, 84), offset_max=25, noise=0.001,
                 decode_attention=True,  **kwargs):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = BaseCNN().to(self.device)
        self.heatmap_head = nn.Conv2d(32, k, 1, 1, 0).to(self.device)
        self.vz_mlp = MLP().to(self.device)

        self.k = k  # Num keypoints.
        self.image_size = image_size  # HxW
        self.heatmap_size = self.encoder.infer_output_size(self.image_size, self.device)  # hxw
        self.epsilon = 1e-6
        self.noise = noise
        self.offset_max = offset_max
        self.decode_attention = decode_attention

        # Setup pixel location buffers.
        r_y = torch.arange(0, self.heatmap_size[0], 1.0)
        r_x = torch.arange(0, self.heatmap_size[1], 1.0)
        rany, ranx = torch.meshgrid(r_y, r_x)
        self.register_buffer("ranx", torch.FloatTensor(ranx).clone().to(self.device))
        self.register_buffer("rany", torch.FloatTensor(rany).clone().to(self.device))

    def downsample_depth(self, rgbd):
        depth = F.interpolate(rgbd[:, 3, :, :].unsqueeze(1), size=self.heatmap_size, mode="bilinear",
                              align_corners=True)
        return depth

    def heatmap_to_xy(self, heatmap):
        heatmap = heatmap / (torch.sum(heatmap, dim=(1, 2), keepdim=True) + self.epsilon)
        sx = torch.sum(heatmap * self.ranx[None], dim=(1, 2))
        sy = torch.sum(heatmap * self.rany[None], dim=(1, 2))
        xy_normalized = torch.stack([sx, sy], 1)
        return xy_normalized

    def heatmap_to_depth(self, heatmap, depth):
        batch_size = depth.shape[0]
        heatmap = heatmap / (torch.sum(heatmap, dim=(1, 2), keepdim=True) + self.epsilon)
        heatmap = heatmap.reshape(batch_size, self.k, self.heatmap_size[0], self.heatmap_size[1])
        keypoint_depth = torch.sum(heatmap * depth.squeeze(1)[:, None, :, :], dim=(2, 3))
        keypoint_depth = keypoint_depth.reshape(batch_size * self.k)
        return keypoint_depth

    def heatmap_to_std(self, heatmap, xy):
        heatmap = heatmap / (torch.sum(heatmap, dim=(1, 2), keepdim=True) + self.epsilon)
        mesh_grid = torch.stack([self.ranx, self.rany], 2)[None]
        var = torch.sum(torch.sum((mesh_grid - xy[:, None, None, :2]) ** 2, 3) * heatmap, dim=(1, 2))
        std = torch.sqrt(var)[:, None]
        return std

    def keypoints_to_heatmap(self, keypoints, heatmap):
        batch_size = keypoints.shape[0]
        keypoints = keypoints.reshape(batch_size * self.k, -1)
        grid = torch.stack([self.ranx, self.rany], 2)[None]  # (b, hs, hs, 2)

        # The variance used for the gaussian map is based on the depth of the keypoint.
        # TODO: This method of calculating variance is arbitrary - is there a better way?
        sigma = (1.0 / (keypoints[:, 2] + self.epsilon)) / 2.0
        var = sigma ** 2.0

        # Calculate gaussian heatmap.
        g_heatmap = torch.exp(-torch.sum((grid - keypoints[:, None, None, :2]) ** 2, 3) / (
                2 * var[:, None, None]))

        if torch.isnan(g_heatmap).any():
            pdb.set_trace()

        # Weight keypoint heatmap by keypoint confidence.
        if self.decode_attention:
            weight = torch.exp(torch.mean(heatmap, dim=(2, 3)))
            weight = weight / (torch.sum(weight, dim=1) + self.epsilon)[:, None]
            weight = weight.reshape(batch_size * self.k)
            g_heatmap = g_heatmap * weight[:, None, None]

        if torch.isnan(g_heatmap).any():
            pdb.set_trace()

        g_heatmap = g_heatmap.reshape(batch_size, self.k, self.heatmap_size[0], self.heatmap_size[1])
        return g_heatmap

    def keypoints_to_3d(self, keypoints, projection_inv, transform_inv, near_val=0.1, far_val=1.0):
        """
        Reproject 2D keypoints (u, v, d) to 3D world space keypoints (x, y, z).
        Does reprojection assuming OpenGL projection matrix (NOT pinhole model).
        Args:
            keypoints (torch.tensor): 2D keypoints (u, v, d) (BxNx3)
            projection_inv (torch.tensor): projection matrix inverse (Bx4x4)
            transform_inv (torch.tensor): transform matrix inverse (Bx4x4)
            near_val (float): OpenGL near val clip
            far_val (float): OpenGL far val clip
        Returns:
            keypoints (torch.tensor): World space keypoints (BxNx3)
        """
        keypoints_hc = torch.ones([keypoints.shape[0], self.k, 4], dtype=torch.float32, device=self.device)
        keypoints_hc[:, :, :3] = keypoints[:, :, :3]
        # Transform 2D keypoints to NDC space.
        keypoints_hc[:, :, 0] = keypoints_hc[:, :, 0] / self.heatmap_size[1]
        keypoints_hc[:, :, 1] = (self.heatmap_size[0] - keypoints_hc[:, :, 1]) / self.heatmap_size[0]
        if torch.isinf(keypoints_hc).any():
            print('keypoints_hc: infi1')
        # print(keypoints_hc[:, :, 2] * (near_val - far_val))
        keypoints_hc[:, :, 2] = (far_val * near_val - far_val * keypoints_hc[:, :, 2]) / (
                keypoints_hc[:, :, 2] * (near_val - far_val))
        if torch.isinf(keypoints_hc).any():
            print(torch.min(keypoints[:, :, 2]))
            print('keypoints_hc: infi2')
        keypoints_hc = (2.0 * keypoints_hc) - 1.0
        # Reproject into camera frame.
        if torch.isnan(keypoints_hc).any():
            print('keypoints_hc: dead2')
        if torch.isinf(keypoints_hc).any():
            print('keypoints_hc: dead3')
        if torch.isnan(projection_inv).any():
            print('projection dead')
        camera_points = projection_inv @ keypoints_hc.transpose(1, 2)
        if torch.isnan(camera_points).any():
            print('camera: dead')
        camera_points = camera_points.transpose(1, 2)

        camera_points = camera_points / camera_points[:, :, 3][:, :, np.newaxis]
        if torch.isnan(camera_points).any():
            print('camera: dead')
        # Transform to world frame.
        world_points = transform_inv @ camera_points.transpose(1, 2)
        world_points = world_points.transpose(1, 2)
        world_points[:, :, 3] = keypoints[:, :, 3]
        if torch.isnan(world_points).any():
            print('3dpoints: dead')
        return world_points

    def heatmap_to_attention(self, heatmap, batch_size):
        heatmap = heatmap.reshape(batch_size, self.k, heatmap.shape[1], heatmap.shape[2])
        avg_score = torch.mean(heatmap, dim=(2, 3))
        avg_score_reg = torch.sum(avg_score, dim=1)
        attention = avg_score / avg_score_reg[:, np.newaxis]
        return attention.reshape(batch_size * self.k)

    def encode(self, rgbd, noise=0.001):
        """
        Encode provided RGBD to set of keypoints.
        Args:
            rgbd (torch.tensor): RGBD input BxHxWx4
        Returns:
            keypoints (torch.tensor): Keypoints
        """
        batch_size = rgbd.shape[0]
        # Get keypoint heatmaps
        z = self.encoder(rgbd)  # BxHxWx4 -> Bxhxwxk
        # print(rgbd[:,3:,:,:])
        if torch.isnan(z).any():
            print('z: dead')
        heatmap = self.heatmap_head(z)
        heatmap = F.softmax(heatmap.reshape(batch_size, self.k, self.heatmap_size[0] * self.heatmap_size[1]), dim=2)
        heatmap = heatmap.reshape(batch_size * self.k, self.heatmap_size[0], self.heatmap_size[1])
        if torch.isnan(heatmap).any():
            print('heatmap: dead')
        # Get keypoints from heatmap.
        keypoints = self.heatmap_to_xy(heatmap)

        # add noise on camera uv plane
        std = self.heatmap_to_std(heatmap, keypoints)
        if noise > 0:
            keypoints = keypoints + std * torch.clamp(torch.randn_like(keypoints) * self.noise, -1.0, 1.0)
        # Get keypoint depth from heatmap and add as third channel.
        if torch.isnan(heatmap).any():
            print('heatmap: dead')
        if torch.isnan(self.downsample_depth(rgbd)).any():
            print('downsample_depth: dead')
        depth = self.heatmap_to_depth(heatmap, self.downsample_depth(rgbd))
        if torch.isnan(depth).any():
            print('depth: dead')
        # Get keypoint attention and add as fourth channel.
        attention = self.heatmap_to_attention(heatmap, batch_size)
        keypoints = torch.cat([keypoints, depth.unsqueeze(1), attention.unsqueeze(1)], dim=1)
        if torch.isnan(keypoints).any() or torch.isnan(heatmap).any() or torch.isnan(attention).any():
            pdb.set_trace()
        return heatmap.reshape(batch_size, self.k, self.heatmap_size[0], self.heatmap_size[1]), keypoints.reshape(batch_size, self.k, 4)

    def forward(self, rgbd, projection_inv, view_inv, noise=None, **kwargs):
        batch_size = rgbd.shape[0]
        if noise is None:
            noise = self.noise

        heatmap, keypoints = self.encode(rgbd, noise=noise)
        keypoints_3d = self.keypoints_to_3d(keypoints, projection_inv, view_inv)
        v_z_hat = keypoints_3d.reshape(batch_size, -1)
        v_z = self.vz_mlp(v_z_hat)
        return v_z

    def loss_forward(self, rgbd, projection_inv, view_inv, noise=None, **kwargs):
        batch_size = rgbd.shape[0]
        if noise is None:
            noise = self.noise

        # Get keypoints.
        heatmap, keypoints = self.encode(rgbd, noise=noise)
        if torch.isnan(keypoints).any():
            print('keypoints: dead')
        keypoints_3d = self.keypoints_to_3d(keypoints, projection_inv, view_inv)
        if torch.isnan(keypoints_3d).any():
            print('keypoints_3d: dead')
        v_z_hat = keypoints_3d.reshape(batch_size, -1)
        if torch.isnan(v_z_hat).any():
            print('v_z_hat: dead')
        v_z = self.vz_mlp(v_z_hat)
        if torch.isnan(v_z).any():
            print('mlp: dead')
        # Get separation loss.
        sep_loss = separation_loss(keypoints_3d[:, :, :3]).mean()

        # Get shift loss.
        s_loss = shift_loss(self, rgbd, self.offset_max, device=self.device)
        ## include alignment later
        return v_z, sep_loss, s_loss



######## image encoder
class BaseCNN(nn.Module):
    def __init__(self, in_channels=4, out_channels=32, n_filters=32, groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters
        self.groups = groups
        self.cnn = self.cnn = self.net = nn.Sequential(
            # (3, 84, 84) -> (42, 42, 42)
            nn.Conv2d(in_channels, 32, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (32, 42, 42) -> (21, 21, 21)
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 21, 21) -> (128, 11, 11)
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 11, 11) -> (256, 6, 6)
            nn.Conv2d(64, out_channels, 3),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.cnn(x)

    def infer_output_size(self, input_size, device=None):
        sample_input = torch.zeros(1, self.in_channels, input_size[0], input_size[1], device=device)
        with torch.no_grad():
            output = self.cnn(sample_input)

        return output.shape[-2], output.shape[-1]



####### MLP
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
        self.net = build_mlp(
            input_dim=input_size,
            output_dim= output_size,
            hidden_units=hidden_sizes,
            hidden_activation=nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)

# # test
# keypoint_net = KeypointNet().to('cuda')
# batch = 10
# sudo_images = torch.ones(batch, 4, 84, 84).to('cuda')
# Inv_Transformation_matrix = torch.tensor([[ 2.58095684e-08,  1.00000000e+00, -1.49011596e-08,  0.00000000e+00],
#                              [-8.66025408e-01,  2.98023224e-08,  4.99999947e-01, -0.00000000e+00],
#                              [ 4.99999947e-01, -1.56300552e-25,  8.66025408e-01,  0.00000000e+00],
#                              [ 6.74999944e-01,  2.50000002e-02,  4.06217835e-01,  1.00000000e+00]]).to('cuda')
#
# Inv_Projection_matrix = torch.tensor([[0.57735028, 0., 0., 0.], [0., 0.57735028, 0., 0.], [ -0., -0., -0., -99.99500128],[0., 0., 1., 100.00499052]]).to('cuda')
# test = keypoint_net.loss_forward(sudo_images, Inv_Projection_matrix, Inv_Transformation_matrix)
# print(test[0].size())