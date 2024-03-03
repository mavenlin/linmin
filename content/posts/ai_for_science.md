---
title: "AI for Science"
author: Min Lin
date: 2024-02-22
summary: " "
katex: true
---

One stereotype of "AI for Science" is to fit transformers on scientific datasets. But that's not the only way AI could benefit science.

In Sea AI Lab we aim to

> **Rebuild density functional theory software with modern tools created in the AI community.**

# Why not blackbox fitting?

I provide a few reasons why the team at Sea AI Lab is currently not working in this direction.

- Just like chemistry people are curious about AI, coming from machine learning background, we're also curious about quantum chemistry, rebuilding DFT from the first principles is a great way to learn.
- Lack of data.
  - At the level of DFT, training blackbox requires data from more accurate sources. DFT studies the electronic structure, which means data from experimental measurements are limited. Anything more accurate than density functional theory would be prohibitively expensive to run at scale.
  - At the level of molecular dynamics, the training data can come from DFT. However, DFT itself is an approximation, with many settings that needs to be done right for different systems.
- Whitebox is always beneficial. The most prominent evidence is that the loss function in DFT calculation is made of kinetic, external, hartree and exchange-correlation energies. We usually keep this structure but only use blackbox models on the exchange-correlation functional.

# What do we do?

## Direct Optimization

At the core of density functional theory is an optimization problem that finds the lowest energy of a system. It is solvable via gradient descent like machine learning models. We enforce the orthogonality constraint in DFT via reparameterization, and implemented energy minimization as gradient descent.

Direct minimization also brings new opportunities. The SCF optimization solves an eigenvalue problem which only works for linear coefficients, while direct optimization enable us to use more complex ansatz (described below). Stochastic gradient is also a very attractive point, where we can replace the integral/summation in DFT with randomized version to save overall computation.

## Automatic differentiation (AD)

Beyond the groundstate energy calculation, various properties of materials are related to the derivatives of the energy. For example, forces are first order derivatives with respect to the atom coordinates. Phonon spectrums are computed from second order derivatives etc. Without AD, the formula of these properties need to be derived by hand and implemented separately. In machine learning, we need to program our own backpropagation algorithm not long ago, but today it is totally reformed by AD tools. We expect AD tools would help transform scientific computing in a similar way.

## Automatic Functional differentiation (AutoFD)

To make it more interesting, we extend AD to support [automatic differentiation of functionals and operators](https://openreview.net/forum?id=gzT61ziSCu). We found it useful for aligning our code with the math derivation. Physicists often communicate in succinct math languages, for example, below is the math for calculating material bandstructure.

1. We first solve the groundstate energy $\rho_0$.
2. We linearize the potential and exchange correlation energy at $\rho_0$ and call it the effective potential.\
$$V_{\text{eff}}=\left.\frac{\delta (E_{\text{pot}}[\rho] + E_{\text{xc}}[\rho])}{\delta \rho}\right\vert_{\rho_0}$$
3. With the effective potential, we then solve the eigenvalue problem \
$$\left(-\frac{1}{2}\nabla^2 + \hat{V}_\text{eff}\right) \psi = \epsilon \psi$$

However, the above is not directly implementable. To implement it, one needs to hand derive the $\frac{\delta E[\rho]}{\delta \rho}$, replace the $\psi$ and $\rho$ with linear combinations of basis, and convert the entire calculation into coefficient space. With the introduction of automatic functional derivative, $\frac{\delta E[\rho]}{\delta \rho}$ is directly implementable as a linearization of the energy functional. We show a code snippet for calculating bandstructure

1. We first define the potential and xc energy functional
   ```python
   def potential_and_xc_energy(rho):
     return (
       e.hartree(rho) +
       e.external(rho, crystal) +
       jax_xc.energy.lda_x(rho)
     )
   ```
2. The $V_\text{eff}$ is the first order derivative of the functional, we can compute it via
   ```python
   Veff = jax.grad(potential_and_xc_energy)(rho0)
   ```
   Since subsequently we use $V_\text{eff}$ in inner products, we could instead directly construct $E_\text{eff}: \rho \mapsto \langle V_\text{eff}|\rho\rangle$
   ```python
   _, Eeff = jax.linearize(potential_and_xc_energy, (rho0,))
   ```
3. We construct the energy under effective potential as $\langle\psi|-\frac{1}{2}\nabla^2+\hat{V}_\text{eff}|\psi\rangle$. We will diagonalize its hessian.
   ```python
   def energy_under_veff(psi):
     return e.kinetic(psi) + Eeff(psi_to_rho(psi))
   ```
4. Numerical computation needs to happen in the parameter space.
   ```python
   def energy_in_param(param, k):
     psi = o.partial(wave_ansatz, args=(param, k), argnums=(0, 1))
     return energy_under_veff(psi)

   bands = {}
   for k in k_vectors:
     bands[k] = jnp.eigh(jax.hessian(energy_in_param)(param, k))[0]
   ```

## More powerful ansatz

In DFT, electron wave functions are approximated as linear combinations of basis functions which are analytically simple functions. When the target function is complex, we increase the number of basis, e.g. we increase the cutoff energy in solid state calculations. One key insight from deep learning is that the increasing the depth is exponentially more efficient than increasing the width.

To introduce deep models as wave functions, we need to consider the following constraints.

- Electron wave functions needs to be a probability magnitude.
- In KS-DFT, different wave functions needs to be orthogonal to each other.
- The energy functionals are made of various integrations, which requires the ansatz to remain efficiently integrable when composed with different functions.

We managed to introduce deep components into the wave functions while satisfying all the above (Section 4 in [DF4T](https://openreview.net/forum?id=aBWnqqsuot7)). We are not the first to explore this idea, [previous work](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.48.11692) dates back to 1993. However, powerful normalizing flows only come to existence in the past few years. It is therefore worthwhile to re-explore these ideas with the tools developed in the AI era.

## Acceleration and scaling

Thanks to the separation of frontend and backend, deep learning frameworks abstract away the details of hardware. Code written in JAX runs on all kinds of hardware without any customization. Today it is even possible to scale out on a cluster transparently via the [parallel jit](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html) api. In our own DFT code, we commit to the idea of separating the main logic from the optimization details. We keep the frontend as simple as math, we hide the optimization and implementation details using the powerful compiler stack in JAX. One such example is the use of FFT in solidstate code. 

The wave functions are represented as linear summation over planewave basis.

$$
\psi(r)=\sum_G c_{G} \exp{iGr}
$$

Although we can evaluate this function explicitly at coordinates to obtain its value, it turns out using _fast fourier transformation_ (FFT) would be much more efficient if we would like to evaluate this function on a grid. Explicit evaluation would take $O(N^2)$ while FFT takes $O(N\log N)$. However, this optimization would break the functional syntax that we're promoting. 

For example, the exchange correlation functional is defined as
$$
E_{\text{xc}}\[\rho\]=\int\epsilon_{\text{xc}}\[\rho\](r)\rho(r)dr
$$

The corresponding python code

```python
Exc = o.integrate(jax_xc.gga_x_pbe(rho) * rho)
```

There are a few difficulties, 

- **Differentiability of** `rho`.\
   gga functionals relies on the first order derivative of the density.\
   `jax_xc.gga_x_pbe` takes `rho` as input evaluates `jax.grad(rho)` internally.\
   Therefore, `rho` has to be a function differentiation w.r.t `r`, while FFT only depends on the linear coefficients.
- **FFT acceleration of the composed function.**\
   Following the functional syntax, we can imagine the internals of a functional to be like the following
   ```python
   def gga_x_pbe(rho):
     def epsilon_xc(r):
       return some_other_function(rho(r), jax.grad(rho(r)))
     return epsilon_xc
   ```
   When the user pass a grid of `r` into the `epsilon_xc` function, it will result in pointwise evaluation of the `rho`, leading to a $O(N^2)$ complexity. Thanks to the internal design of JAX, we can register customized _jvp rule_ and _batch rule_ for wave functions that are composed of linearly combined planewaves. Via batch rule, we can trigger the FFT acceleration when a grid of input is received, while at the same time we write custom jvp rules to guarantee that the gradient is still computed correctly.
