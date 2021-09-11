import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision import utils as vutils

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, 
                dim, 
                window_size, 
                num_heads, 
                dilated=False, 
                qkv_bias=True, 
                qk_scale=None, 
                attn_drop=0., 
                proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.dilated = dilated

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def Q(self, x):
        if self.training:
            return x + torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
        else:
            return torch.round(x)

    def forward(self, ref, adj, x_shape, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = ref.shape
        B, H, W = x_shape
        q = self.q(ref).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(adj).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple) (B, heads, N, C//heads)
        if self.dilated:
            q = q[:,:,::2,:]
            k = k[:,:,::2,:]

        q = q * self.scale
        
        attn = (q @ k.transpose(-2, -1))
        if self.dilated:
            attn = F.interpolate(attn, size=N, mode='bilinear')

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        nW = H // self.window_size * W // self.window_size
        if mask is not None:
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.to(attn.device).unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn) # [b_, num_heads, N, N]

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, dilated=False, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.dilated = dilated
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim, 
            window_size=self.window_size, 
            num_heads=num_heads,
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop, 
            proj_drop=drop, 
            dilated=dilated)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, ref, adj, mask_matrix, x_shape):
        B, L, C = ref.shape
        H, W = x_shape
        assert L == H * W, "input feature has wrong size"

        shortcut = ref
        #shortcut = adj # decoder cannot get x
        ref = self.norm1(ref)
        adj = self.norm1(adj)
        ref = ref.view(B, H, W, C)
        adj = adj.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        ref = F.pad(ref, (0, 0, pad_l, pad_r, pad_t, pad_b))
        adj = F.pad(adj, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = ref.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_ref = torch.roll(ref, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_adj = torch.roll(adj, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_ref = ref
            shifted_adj = adj
            attn_mask = None

        # partition windows
        ref_windows = window_partition(shifted_ref, self.window_size)  # nW*B, window_size, window_size, C
        ref_windows = ref_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        adj_windows = window_partition(shifted_adj, self.window_size)  # nW*B, window_size, window_size, C
        adj_windows = adj_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        x_windows = self.attn(ref_windows, adj_windows, x_shape=(B,H,W), mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        shifted_x = x_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(shifted_x, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        #x = self.drop_path(self.mlp(self.norm2(x)))
        return x


class MaskLayer(nn.Module):
    def __init__(self,
                 patch_size,
                 window_size,
                 dilated=False):

        super().__init__()
        self.patch_size = patch_size
        if dilated:
            window_size = window_size * 2 - 1
        self.window_size = window_size
        self.shift_size = window_size // 2

    def forward(self, H, W):
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.patch_size))
        Wp = int(np.ceil(W / self.patch_size))
        Hw = int(np.ceil(Hp / self.window_size)) * self.window_size
        Ww = int(np.ceil(Wp / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hw, Ww, 1))  # 1 Hw Ww 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask, Hp, Wp


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=64, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        Hp = int(np.ceil(H / self.patch_size)) * self.patch_size
        Wp = int(np.ceil(W / self.patch_size)) * self.patch_size
        x = F.pad(x, (0, Wp - W))
        x = F.pad(x, (0, 0, 0, Hp - H))

        x = self.proj(x)  # B C Hp Wp
        Hp, Wp = x.size(2), x.size(3)
        if self.norm is not None:
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Hp, Wp)

        return x


class PatchRec(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, patch_size=4, in_chans=64, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, H, W):
        B, L, C = x.shape
        Hp = int(np.ceil(H / self.patch_size))
        Wp = int(np.ceil(W / self.patch_size))
        x = self.proj(x.transpose(1, 2).view(B,C,Hp,Wp))
        return x[:,:,:H,:W]


class SwinTransformerAlignment(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        layers (list): [n_layer(int), heads(int), patch(int), window(int), dilated(bool)]
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, 
                 in_chans=64,
                 embed_dim=96, 
                 layers_cfg=[4,3,2,10,False], # ban dilated
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None,
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, 
                 patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        # use 1 stage now
        assert type(layers_cfg[0]) is not list, "accept single stage only"

        self.embed_dim = embed_dim
        (self.depth, self.num_heads, self.patch_size, self.window_size, self.dilated) = layers_cfg
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule

        # build layers
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.mask_maker = MaskLayer(
            patch_size=self.patch_size,
            window_size=self.window_size,
            dilated=False)

        self.patch_rec = PatchRec(
            patch_size=self.patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim)

        self.blocks  = nn.ModuleList([
            SwinTransformerBlock(
                dim=embed_dim, 
                num_heads=self.num_heads, 
                window_size=self.window_size,
                shift_size=0 if (j % 2 == 0) else self.window_size // 2,
                dilated=self.dilated,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[j] if isinstance(dpr, list) else dpr,
                norm_layer=norm_layer)
            for j in range(self.depth)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, ref, x): # let ref -> x
        _, _, h, w = x.shape
        ref_embed = self.patch_embed(ref)
        ref_embed = self.pos_drop(ref_embed.flatten(2).transpose(1, 2))
        x_embed = self.patch_embed(x)
        x_embed = self.pos_drop(x_embed.flatten(2).transpose(1, 2))
        mask, Hp, Wp = self.mask_maker(h, w)

        for j in range(self.depth):
            if self.use_checkpoint:
                ref_embed = checkpoint.checkpoint(self.blocks[j], x_embed, ref_embed, mask, (Hp, Wp))
            else:
                ref_embed = self.blocks[j](x_embed, ref_embed, mask, (Hp, Wp))

        aligned_ref = self.patch_rec(ref_embed, h, w)

        return aligned_ref