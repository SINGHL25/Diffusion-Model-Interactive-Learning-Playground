def get_ddpm_math_explanation():
    return """
    **DDPM (Denoising Diffusion Probabilistic Models)** follow a two-step process:

    1.  **Forward Diffusion (Noising Process):**
        *   We start with a clean image $x_0$.
        *   Gradually add Gaussian noise over $T$ timesteps, creating a sequence $x_1, x_2, ..., x_T$.
        *   The distribution $q(x_t | x_0)$ is a Gaussian where the mean and variance are controlled by a fixed schedule of $\beta_t$ values.
        *   $q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\\bar{\\alpha}_t} x_0, (1 - \\bar{\\alpha}_t) \mathbf{I})$
        *   Where $\\alpha_t = 1 - \\beta_t$ and $\\bar{\\alpha}_t = \prod_{s=1}^t \\alpha_s$.
        *   A key aspect is that $x_t$ can be directly sampled from $x_0$ using the reparameterization trick:
            $x_t = \sqrt{\\bar{\\alpha}_t} x_0 + \sqrt{1 - \\bar{\\alpha}_t} \epsilon$, where $\epsilon \sim \mathcal{N}(0, \mathbf{I})$.

    2.  **Reverse Diffusion (Denoising Process):**
        *   This is the learning part. We train a neural network (typically a U-Net) to predict the noise $\epsilon$ added at each step.
        *   The model learns to approximate $p_\theta(x_{t-1} | x_t)$, the reverse of the forward process.
        *   The reverse step is also Gaussian, and its mean and variance depend on the predicted noise.
        *   The objective is to minimize the **variational lower bound (VLB)** on the negative log-likelihood of the data.
        *   The simplified objective directly optimizes the prediction of $\epsilon$: $\mathbb{E}_{t, x_0, \epsilon} [\| \epsilon - \epsilon_\theta(x_t, t) \|^2]$.
    """

def get_ddim_math_explanation():
    return """
    **DDIM (Denoising Diffusion Implicit Models)** builds upon DDPM but introduces a more generalized sampling process.

    *   **Non-Markovian Forward Process:** Unlike DDPM's fixed Markovian forward process, DDIM can define a non-Markovian forward process.
    *   **Deterministic Sampling:** The key innovation is to make the reverse sampling process deterministic (when $\eta=0$). This means that given $x_t$, the estimate for $x_0$ (denoted $\hat{x}_0$) and then $x_{t-1}$ are uniquely determined without adding new random noise.
    *   **Generalized Reverse Step:**
        $x_{t-1} = \sqrt{\\bar{\\alpha}_{t-1}} \hat{x}_0 + \sqrt{1 - \\bar{\\alpha}_{t-1} - \\sigma_t^2} \epsilon_\theta(x_t, t) + \\sigma_t \mathbf{z}$
        Where:
        *   $\hat{x}_0 = \frac{x_t - \sqrt{1 - \\bar{\\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\\bar{\\alpha}_t}}$ (predicted original image)
        *   $\\sigma_t^2 = \\eta^2 \frac{1 - \\bar{\\alpha}_{t-1}}{1 - \\bar{\\alpha}_t} (1 - \frac{\\bar{\\alpha}_t}{\\bar{\\alpha}_{t-1}})$ (variance, controlled by $\\eta$)
        *   $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ (random noise, only if $\\eta > 0$)

    *   **Faster Sampling:** By making the reverse process deterministic, DDIM allows for generating high-quality samples with significantly fewer steps than DDPM (e.g., 50-100 steps instead of 1000).
    *   **Coherence and Inversion:** The deterministic nature enables better latent space interpolation and also allows for a more direct "inversion" of the diffusion process (finding the latent code for a given image).
    """
