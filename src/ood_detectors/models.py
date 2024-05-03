import math
import torch
import torch.nn as nn
from einops import repeat


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :param repeat_only: if True, only repeat the timesteps without embedding.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    :param module: the module to zero out.
    :return: the zeroed module.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param mid_channels: the number of middle channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param use_context: if True, use context conditioning.
    :param context_channels: the number of context channels.
    """

    def __init__(
        self,
        channels,
        mid_channels,
        emb_channels,
        dropout,
        use_context=False,
        context_channels=512
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout

        self.in_layers = nn.Sequential(
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Linear(channels, mid_channels, bias=True),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, mid_channels, bias=True),
        )

        self.out_layers = nn.Sequential(
            nn.LayerNorm(mid_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Linear(mid_channels, channels, bias=True)
            ),
        )

        self.use_context = use_context
        if use_context:
            self.context_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(context_channels, mid_channels, bias=True),
            )

    def forward(self, x, emb, context):
        """
        Forward pass of the residual block.
        :param x: the input tensor.
        :param emb: the timestep embedding tensor.
        :param context: the context tensor.
        :return: the output tensor.
        """
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        if self.use_context:
            context_out = self.context_layers(context)
            h = h + emb_out + context_out
        else:
            h = h + emb_out
        h = self.out_layers(h)
        return x + h


class SimpleMLP(nn.Module):
    """
    The full skip network with timestep embedding.
    :param channels: channels in the input Tensor.
    :param time_embed_dim: dimension of the timestep embedding.
    :param bottleneck_channels: base channel count for the model.
    :param num_res_blocks: number of residual blocks per downsample.
    :param dropout: the rate of dropout.
    :param use_context: if True, use context conditioning.
    :param context_channels: the number of context channels.
    """

    def __init__(
        self,
        channels,
        time_embed_dim=256,
        bottleneck_channels=1536,
        num_res_blocks=13,
        dropout=0,
        use_context=False,
        context_channels=512
    ):
        super().__init__()

        self.image_size = 1
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout

        self.time_embed = nn.Sequential(
            nn.Linear(channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_proj = nn.Linear(channels, channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                channels,
                bottleneck_channels,
                time_embed_dim,
                dropout,
                use_context=use_context,
                context_channels=context_channels
            ))

        self.res_blocks = nn.ModuleList(res_blocks)

        self.out = nn.Sequential(
            nn.LayerNorm(channels, eps=1e-6),
            nn.SiLU(),
            zero_module(nn.Linear(channels, channels, bias=True)),
        )

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        x = x.squeeze()
        x = self.input_proj(x)
        t_emb = timestep_embedding(timesteps, self.channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        for block in self.res_blocks:
            x = block(x, emb, context)
        x = self.out(x)
        return x
