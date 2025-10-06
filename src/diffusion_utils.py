import torch

def get_time_embedding(timesteps, embedding_dim):
    """
    Creates a sinusoidal positional embedding for timesteps.
    Used to inject time information into the UNet.
    """
    # From "Attention Is All You Need"
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1: # pad with a zero if odd
        emb = torch.cat([emb, torch.zeros(timesteps.shape[0], 1)], dim=1)
    return emb

# Other utility functions like noise schedules can also live here if not in DDPM class.
