---
title: "Philosophical bits that I collected over the years for building AI."
author: Min Lin
date: 2024-03-03
summary: " "
katex: true
---

- **Brains are ODE/SDE**\
  The neurons of human brains are interconnected variables, they can be described as a set of ODEs, or stochastic differential equations (SDE) considering the noise. Normally we would think of a parameterized function to govern the evolution of neurons, i.e. we learn and fix the connection weights. However, for the sake of a general learner, we would want to describe the learning rules too! Once we include the connection weights as variables in our SDE, the function that governs our system needs to be designed and prefixed. Namely, the General Learner described in my [research page](/research/) needs to be a human designed rule without free parameters.

- **The ODE/SDE must operate in two modes, with input and without input.**\
  The former is easy to understand, we must store the information in our observations in the variables of the SDE. The latter make sense because we learn from our meditation without input. We don't get mad or lose memory in a short period of time without information input, therefore, our model needs to stay sane too even when evolved in time without any input.
  
- **Critical points must play a role in intelligence**\
  Just like hopfield networks utilize attractors, I believe learning has something to do with the critical points in the ODE/SDE.
  - Symbolic reasoning capability of human beings must have something to do with the critical points. Symbols are discrete in nature, it is a very natural way to induce discrete structures from continuous models.
  - Currently we believe we can represent concepts as vectors in a continuous space and use learned continuous function to process these vectors. This aligns with the ODE view because we can see ODE as a recurrent network operating on a vector. However, when the weights of the function change a little bit, the resulting vector would drift a bit. This is unacceptable when our brain is a ODE/SDE that will roll forward in time, because even if every step a very small amount of our neural weights are changed, it would have caused drastic change over time, and the meaning of these vector would be completely different. It is likely that to reliably store information, we need to make use of structures like attractors, which has a discrete nature.
  - Language models need to project the embedding back to the word space (discretization) before it is input back to the network. Otherwise if the output embedding is directly fed back to the input, it would go abberant very quickly.


- **The ODE/SDE needs to be a probability model implicitly**\
  Because there are so many evidence that human being has a super powerful probability model of the world. It doesn't necessarily mean that we can generate images like stable diffusion. But we definitely have a probability model to tell images that aren't natural, we are super hard to fool. Therefore, the ODE/SDE must contain an implicit probability model. It is just we can't sample from it like stable diffusion, as the computation theory says, to verify is easier than to solve.

> I've been thinking what kind of rule it is that governs our internal ODE/SDE. Existing works satisfy some of the above points,
> 1.  Learning rules from biological inspirations, for example, [Hebbian rule](https://en.wikipedia.org/wiki/Hebbian_theory), [STDP](https://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity). They describe learning rules, aka, both neuron activations and the connection weights are part of the ODE/SDE variables, and the evolution function is the learning rule. They lack mathematical explaination what on earth is done via these rules.
> 2. [Associative memory](https://en.wikipedia.org/wiki/Autoassociative_memory) using hopfield net. Attractors are used to store information, it is also a learning rule, but it lacks the probability part.
> 3. BPTT, if we see RNN as a discretized ODE, BPTT is a pragmatic algorithm to train RNN. However, none of the above philosophical points are satisfied.
> 4. [Neural ODE as normalizing flow](https://arxiv.org/abs/1810.01367), and [diffusion models](https://arxiv.org/abs/2011.13456). Both of them create a probability model out of ODE/SDE. But they don't make use of critical points (Both of them are evolve the system for a fixed time window from a prefixed distribution to the target distribution. Nothing is said about what if the system is evolve further in time.), and they are not about learning rules.

- **Number of latent variables should be much larger than the input**\
  Example: our brains, therefore architectures that reduce dimension is not quite right.

- **Training / Inference trade-off**\
  (Here training / inference don't mean the two phases that we commonly refer to in deep learning. Training means learning information from observations, inference means using the information for making predictions, in an online agent, these don't have to be in two phases.)
  Learning is about storing information of the observations. The storage happens in a format such that it is easy to use the existing information for making predictions. There is always a trade-off between storing the information and using it. **If we choose the storage format to be raw data**, learning would be super easy and fully online, we just append new observations to the dataset. But using is hard, for example, we could train a neural network on the dataset and then use the neural network to make predictions. Or if we use kernel machine, it would take $O(N^2)$ where $N$ is the number of observations we already collect. **If we choose neural network weights** as the storage format, then using it is super easy, make a prediction is $O(1)$. However, learning becomes hard, for every new observation added, we have to retrain the neural network on all data collected so far. The key to online learning must be finding a sweet spot on this trade-off.

- **We may not need generating models, but we need generative models.**\
  Yann LeCun is [against generating pixels](https://x.com/ylecun/status/1759486703696318935), while there is a subtle difference between a **generative model**(having a probability model) and a **generating model**(being able to sample). I agree on the generating pixel is wasteful part, but disagree with the more general dissing on generative models.
  My belief comes from the fact that human beings are the golden standard to evaluate generated images/videos. Although only professional painters are able to paint well, every normal human can serve as an evaluator on whether the generated image from a model is realistic. Meaning every human knows exactly where the data should be in the high dimensional pixel space. Meaning human internally have a good probability model of the pixel space observation, better than any SOTA generating models.
