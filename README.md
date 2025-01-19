Provided are GAN losses from the [R3GAN](https://arxiv.org/abs/2501.05441) and [Seaweed](https://arxiv.org/abs/2501.08316) papers.

## Relativistic GAN loss referenced in the R3GAN paper:

$`{L}(\theta, \psi) = \mathbb{E}_{z \sim p_z, \, x \sim p_D} \left[ f \left( D_\psi(G_\theta(z)) - D_\psi(x) \right) \right]`$


### 1. Default logistic GAN loss

$`f(t) = -\log(1 + e^{-t})`$

```python
# If f is omitted, logistic_f is used automatically
d_loss = gan_loss(discriminator, generator, real_images, z)

# You have to freeze the other model when doing a backward pass for generator or discriminator, otherwise you will combine the negative and positive gradients which will cancel out.
g_loss = -gan_loss(discriminator, generator, real_images, z)
```

### 2. Custom function 𝑓 If you have a different function (e.g. hinge), you can pass it in:
```python
def hinge_f(t):
    return torch.nn.functional.relu(1 - t)

d_loss = gan_loss(
    discriminator, generator, real_images, z,
    f=hinge_f
)
```

### 3. Extra arguments to discriminator or generator, if your models need additional inputs:
```python
d_loss = gan_loss(
    discriminator, generator,
    real_images, z,
    generator_args=(some_label, ),          # positional
    generator_kwargs={'some_flag': True},   # keyword
    disc_args=(some_label, ),
    disc_kwargs={'cond_flag': True}
)
```

## R1 and R2 losses referenced in the R3GAN paper:

$`R_1(\psi) = \frac{\gamma}{2} \mathbb{E}_{x \sim p_D} \left[ \| \nabla_x D_\psi \|^2 \right]`$

$`R_2(\theta, \psi) = \frac{\gamma}{2} \mathbb{E}_{x \sim p_\theta} \left[ \| \nabla_x D_\psi \|^2 \right]`$

### Use the 0 centred gradient penalties to stabilize training:
```python
d_loss += r1_penalty(discriminator, real_images, gamma=1.0, disc_args=args, disc_kwargs=kwargs) 
d_loss += r2_penalty(discriminator, fake_images, gamma=1.0, disc_args=args, disc_kwargs=kwargs)
```

## Approximate R1 loss referenced in the Seaweed paper, and a extrapolated version of the R2 loss:
Here is the LaTeX representation of \(\mathcal{L}_{aR1}\) and \(\mathcal{L}_{aR2}\) with expectation notation for \(p_D\) and \(p_\theta\), respectively:


$`{L}_{aR1} = \lambda * \mathbb{E}_{x \sim p_D} \left[ \left\| D(x, c) - D\big(\mathcal{N}(x, \sigma I), c\big) \right\|_2^2 \right`]`$

$`{L}_{aR2} = \lambda * \mathbb{E}_{x \sim p_\theta} \left[ \left\| D(x, c) - D\big(\mathcal{N}(x, \sigma I), c\big) \right\|_2^2 \right]`$


The idea is that, the gradient penalty exists to disencourage large changes in the logits from a small change in the input. The approximation just adds a bit of noise to the input, which works similarly to taking the gradient in this case. This approximation is very helpful as FlashAttention cannot take a second derivative, so the true penalties are much slower to compute.

### Use the 0 centred approximate gradient penalties to stabilize training:
```python
d_loss += approximate_r1_loss(discriminator, real_images, sigma=0.01, Lambda=100.0, disc_args=disc_args, disc_kwargs=disc_kwargs) 
d_loss += approximate_r2_loss(discriminator, fake_images, sigma=0.01, Lambda=100.0, disc_args=disc_args, disc_kwargs=disc_kwargs)
```