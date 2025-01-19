import torch

####################################
# Default logistic function
####################################
def logistic_f(t):
    """
    f(t) = -log(1 + e^-t)
    """
    return -torch.log(1 + torch.exp(-t))


####################################
# 1) Main GAN Loss with default f(t)
####################################
def gan_loss(
    discriminator,
    generator,
    real_images,
    z,
    f=None,                 # If None, defaults to logistic_f
    generator_args=None,
    generator_kwargs=None,
    disc_args=None,
    disc_kwargs=None
):
    """
    Generic GAN loss of the form:
        E_{z,x}[ f( D(G(z)) - D(x) ) ].

    Args:
        discriminator: Discriminator network D
        generator: Generator network G
        real_images: A batch of real data (x)
        z: Noise tensor sampled from p_z
        f: Callable for f(D(fake) - D(real)) [defaults to logistic_f]
        generator_args: Extra positional args for G
        generator_kwargs: Extra keyword args for G
        disc_args: Extra positional args for D
        disc_kwargs: Extra keyword args for D

    Returns:
        A scalar tensor for the GAN loss.
    """
    # Default the function to logistic_f if none is provided
    if f is None:
        f = logistic_f

    if generator_args is None:
        generator_args = ()
    if generator_kwargs is None:
        generator_kwargs = {}
    if disc_args is None:
        disc_args = ()
    if disc_kwargs is None:
        disc_kwargs = {}

    # Generate fake images
    fake_images = generator(z, *generator_args, **generator_kwargs)

    # Evaluate discriminator
    disc_real = discriminator(real_images, *disc_args, **disc_kwargs)
    disc_fake = discriminator(fake_images, *disc_args, **disc_kwargs)

    # Compute the loss using default or provided f
    loss = f(disc_fake - disc_real).mean()
    return loss


####################################
# 2) Exact R1 Penalty
####################################
def r1_penalty(
    discriminator,
    real_images,
    gamma=1.0,
    disc_args=None,
    disc_kwargs=None
):
    """
    R1 penalty:
        R1(psi) = gamma/2 * E_{x ~ p_D}[ || grad_x D(x) ||^2 ]
    """
    if disc_args is None:
        disc_args = ()
    if disc_kwargs is None:
        disc_kwargs = {}

    real_images = real_images.clone().detach().requires_grad_(True)
    real_scores = discriminator(real_images, *disc_args, **disc_kwargs).sum()

    # Grad of sum of outputs w.r.t. inputs
    grads = torch.autograd.grad(
        outputs=real_scores,
        inputs=real_images,
        create_graph=True
    )[0]

    # L2 norm of gradients per sample
    grad_penalty = grads.view(grads.size(0), -1).pow(2).sum(dim=1).mean()
    return 0.5 * gamma * grad_penalty


####################################
# 3) Exact R2 Penalty
####################################
def r2_penalty(
    discriminator,
    fake_images,
    gamma=1.0,
    disc_args=None,
    disc_kwargs=None
):
    """
    R2 penalty:
        R2(theta, psi) = gamma/2 * E_{x ~ p_theta}[ || grad_x D(x) ||^2 ]
    """
    if disc_args is None:
        disc_args = ()
    if disc_kwargs is None:
        disc_kwargs = {}

    fake_images = fake_images.clone().detach().requires_grad_(True)
    fake_scores = discriminator(fake_images, *disc_args, **disc_kwargs).sum()

    grads = torch.autograd.grad(
        outputs=fake_scores,
        inputs=fake_images,
        create_graph=True
    )[0]

    grad_penalty = grads.view(grads.size(0), -1).pow(2).sum(dim=1).mean()
    return 0.5 * gamma * grad_penalty


####################################
# 4) Approximate R1 Loss
####################################
def approximate_r1_loss(
    discriminator,
    real_images,
    sigma=0.01,
    Lambda=100.0,
    disc_args=None,
    disc_kwargs=None
):
    """
    Approx. R1 via finite differences:
        L_{aR1} = || D(x) - D(x + noise) ||^2
    """
    if disc_args is None:
        disc_args = ()
    if disc_kwargs is None:
        disc_kwargs = {}

    noise = sigma * torch.randn_like(real_images)
    d_real = discriminator(real_images, *disc_args, **disc_kwargs)
    d_noisy = discriminator(real_images + noise, *disc_args, **disc_kwargs)
    return ((d_real - d_noisy).pow(2).mean()) * Lambda


####################################
# 5) Approximate R2 Loss
####################################
def approximate_r2_loss(
    discriminator,
    fake_images,
    sigma=0.01,
    Lambda=100.0,
    disc_args=None,
    disc_kwargs=None
):
    """
    Approx. R2 via finite differences:
        L_{aR2} = || D(x_fake) - D(x_fake + noise) ||^2
    """
    if disc_args is None:
        disc_args = ()
    if disc_kwargs is None:
        disc_kwargs = {}

    noise = sigma * torch.randn_like(fake_images)
    d_fake = discriminator(fake_images, *disc_args, **disc_kwargs)
    d_fake_noisy = discriminator(fake_images + noise, *disc_args, **disc_kwargs)
    return ((d_fake - d_fake_noisy).pow(2).mean()) * Lambda
