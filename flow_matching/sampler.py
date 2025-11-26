from torch import Tensor

def expand_t_like_x(t: float | Tensor, x: Tensor) -> Tensor:
    """Expand time vector t to match the shape of tensor x without making a copy.

    Args:
        t (float | Tensor): 1d tensor with shape (batch_size,).
        x (Tensor): Any tensor with shape (batch_size, ...).

    Returns:
        Tensor: (batch_size, ...)

    Examples:
        >>> expand_t_like_x(0.5, torch.randn(10, 1, 28, 28)) # 0.5
        >>> expand_t_like_x(torch.rand(10), torch.randn(10, 1, 28, 28)) # (10, 1, 28, 28)
    """

    if isinstance(t, float):
        # We do not expand scalar values
        return t

    assert t.ndim == 1, "Time vector t must be a 1d tensor with shape (batch_size,)."
    assert t.size(0) == x.size(0), "Time vector t must have the same batch size as x."

    return t.reshape(-1, *([1] * (x.ndim - 1))).expand_as(x)

class PathSampler:
    def __init__(self, sigma_min: float = 0.0):
        # Minimal Gaussian distribution std for target distribution
        self.sigma_min = sigma_min

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Samples points x_t from the distribution p_t(x_t | x_1)

        According to eq. (20) in the paper (https://arxiv.org/abs/2210.02747), we have:

            mu_t = t * x_1, and sigma_t = 1 - (1 - sigma_min) * t

        and the conditonal flow phi_t(x_0) is given by (see eq. 22):

            phi_t(x_0) = mu_t + sigma_t * x_0

        We treat x_t as the flow of phi_t(x_0), then we have:

            x_t = t * x_1 + (1 - (1 - sigma_min) * t) * x_0

        And, the target conditional vector field u_t(x_1|x_0) equals to the derivative of x_t w.r.t t:

            u_t(x_1|x_0) = dx_t = x_1 - (1 - sigma_min) * x_0

        We can use this to construct the conditional flow matching loss:

            loss = loss_fn(vf_t(x_t), dx_t)

        This function returns the samples x_t ~ p_t(x_t | x_1) and dx_t to compute the above loss.

        Args:
            x_0: Samples from base distribution, (batch_size, ...)
            x_1: Samples from target distribution, (batch_size, ...)
            t: Time steps, (batch_size,)

        Returns:
            Tensor: Samples from the distribution p_t(x_t | x_1), (batch_size, ...)
        """

        t = expand_t_like_x(t, x_1)
        mu_t = t * x_1
        sigma_t = 1 - (1 - self.sigma_min) * t

        # Draw a sample from the probability path: x_t ~ p_t(x|x_1)
        x_t = mu_t + sigma_t * x_0

        # Compute the target conditional vector field
        dx_t = x_1 - (1 - self.sigma_min) * x_0

        return x_t, dx_t