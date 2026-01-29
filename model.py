
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_hidden, n_heads=4, ffn_ratio=4.0, dropout=0.2, albi_mask=None):

        super().__init__()

        self.num_heads = n_heads
        self.albi_mask = albi_mask

        self.attention = nn.MultiheadAttention(
            embed_dim=d_hidden, num_heads=n_heads, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_hidden, int(d_hidden * ffn_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_hidden * ffn_ratio), d_hidden),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(
            norm_x, norm_x, norm_x, attn_mask=self.albi_mask)
        x = x + attn_out  # residual, we learn f(x) - x
        x = x + self.ffn(self.norm2(x))

        return x


class POSTransformer(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size, num_pos, positional_layer,  d_hidden=128, dropout=0.2, ffn_ratio=4.0):
        super().__init__()
        # lets tokenize first (apply the vocab mapping)

        self.embedding = nn.Embedding(vocab_size, d_hidden)
        self.use_ALiBi = (
            positional_layer is isinstance(positional_layer, AlBI))

        self.bias_mask = None
        if self.use_ALiBi:
            self.bias_mask = positional_layer.bias

        self.positional_layer = positional_layer

        blocks = []

        for _ in range(num_layers):
            blocks.append(TransformerBlock(
                d_hidden=d_hidden, n_heads=num_heads, ffn_ratio=ffn_ratio, dropout=dropout, albi_mask=self.bias_mask))
        self.blocks = nn.Sequential(*blocks)

        # prediction head

        self.prediction_head = nn.Linear(d_hidden, num_pos)

    def forward(self, x):

        x = self.embedding(x)  # (batch, seq, hidden_d)

        x = self.positional_layer(x)
        x = self.blocks(x)

        logits = self.prediction_head(x)
        return logits


class AlBI(nn.Module):
    def __init__(self, num_heads, seq_len, device):
        super().__init__()
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.device = device
        self.bias = self._precompute_alibi()

    def forward(self, X):
        return X

    def _precompute_alibi(self):
        """
        Basically want to create a mask that adds m * -(i-j)
        the m is based on the head value (2^-8/H) - the heads 
        """

        slopes = torch.tensor(
            [2**(-8 / i) for i in range(1, self.num_heads + 1)]).reshape(self.num_heads, 1, 1)

        q_position = torch.arange(
            self.seq_len, dtype=torch.float32)[:, None]  # :none adds a new dimension, this makes it a column vector

        k_position = torch.arange(
            self.seq_len, dtype=torch.float32)[None, :]  # none adds a new dimension making it a row vector

        change = k_position - q_position

        bias = slopes * change.unsqueeze(0)  # unsqueeze adds a third dimension

        bias.to(self.device)
        return bias


class RoPE(nn.Module):

    def __init__(self, dim, length):
        super().__init__()
        self.dim = dim
        self._precompute_freq(dim, length)

    def _precompute_freq(self, dim, end, theta=10000.0):
        """Algo is simple, 
        we define some base, then we calculate a bunch of frequencies
        first couple dimensions rotate slowly while others rotate a lot

        we multply the rotation * sequence position (uniform rotation)

        instead of using a rotation matrix we use euelers formula
        multiplying two complex numbers creates a rotation!
        we treat each (x1,x2) as a complex number itself for the ortation
        this is the exact same as the 2d rotation matrix
        """

        # this is our frequency at every pair of (x_1, x_2), (x_3, x_4)

        # Math: theta_i = 10000^(-2i/d)
        thetas = 1.0 / (theta ** (torch.arange(0, dim, 2)
                        [: (dim // 2)].float() / dim))
        positions = torch.arange(end)
        freqs = torch.outer(positions, thetas).float()

        # Math: e^(i * m * theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        # register_buffer ensures this moves with model.to('cuda')
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x):
        """
        View each pair as complex,
        then rotate 
        """
        seq_len = x.shape[1]

        if seq_len > self.freqs_cis.shape[0]:
            self._precompute_freq(self.dim, seq_len)

        freqs_cis = self.freqs_cis[:seq_len].view(1, seq_len, -1)
        x_complex = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], -1, 2))

        x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
        return x_rotated.type_as(x)


class LearnedEmbeddings(nn.Module):

    def __init__(self, max_len, d_model, device):
        super().__init__()

        self.embeddings = nn.Embedding(max_len, d_model)
        self.range = torch.arange(max_len).to(device)

    def forward(self, x):
        embeddings = self.embeddings(self.range[:x.shape[1]])
        x = x + embeddings
        return x
