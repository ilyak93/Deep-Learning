r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0, seq_len=0,
        h_dim=0, n_layers=0, dropout=0,
        learn_rate=0.0, lr_sched_factor=0.0, lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 256
    hypers['seq_len'] = 64
    hypers['h_dim'] = 1024
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.25
    hypers['learn_rate'] = 0.001
    hypers['lr_sched_factor'] = 0.5
    hypers['lr_sched_patience'] = 2
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    temperature = 0.4
    start_seq = 'Act I: To be or not to be, is that the question?'
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=16,
        h_dim=256, z_dim=128, x_sigma2=0.01,
        learn_rate=0.001, betas=(0.9, 0.999),
    )
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 32
    hypers['z_dim'] = 100
    hypers['data_label'] = 1
    hypers['label_noise'] = 0.2
    hypers['discriminator_optimizer']['type'] = 'SGD'
    hypers['discriminator_optimizer']['lr'] = 0.002
    #hypers['discriminator_optimizer']['betas'] = (0.5, 0.999)
    hypers['generator_optimizer']['type'] = 'Adam'
    hypers['generator_optimizer']['lr'] = 0.0002
    hypers['generator_optimizer']['betas'] = (0.5, 0.999)
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


When training one component we assume that the other one is given. Meaning that while training the Discriminator we assume
a given generator, and while training the generator we assume a given discriminator.
Thus, during training, when training the discriminator specifically, we want to discard the gradients when sampling from
the GAN, because we don't want the gradients to backpropagate through the discriminator loss, so it will have an 
effect on the weights of the generator, altough we wish to train the discriminator and the generator seperatly.
When we train the generator, we maintain the gradients as the sampler belong to the generator, and we want to
modify the generator weights according to the sampled vector.

"""

part3_q2 = r"""
**Your answer:**


1.  No. Since we test/train every component (discriminator/generator) alone, meaning we assume that the other one is given,
    it is not enough to have a generator with a low loss value if our discriminator is with very high loss value, meaning it's
    not accurate. Since the discriminator is not accurate, the loss and accuracy for the generator is not accurate as well!
    In other words: we can get a really good results based on a bad discriminator which basically says that this score is
    useless and does not reflect reality.

2.  When the generator's loss is becoming smaller, meaning that the generator becomes better, we expect the discriminator
    to improve as well since its performance depends on how well the generator is. When the generator improves but the
    discriminator is not we can conclude that the discriminator does not perform well and the generator uses this fact
    to learn how to fool it to predict similarities even when it shouldn't.

"""

part3_q3 = r"""
**Your answer:**


As can be seen from the results, the images generated from the VAE seems more blurred than those generated from
the GAN, but on the other hand - they seem more 'real' and they are better in perceptation of the general and undetailed shapes,
as in the GAN we can observe a lot of deformations and unrealistic patches. In the VAE images as was allready mentioned in the previous 
question section, we could control the variance, and if we would take a variance of 0.01 we could even better see the general form Bush's 
face, but still the VAE images have not good details. 
The GAN's images are sharper and the face features are clearer, i.e there are some times
we could better recognize the face impression or see details never seen on the VAE images, such as mouth with teeth or
more accurate look of the eyes and e.t.c.
These changes are related mainly to the loss differences. While in the VAE, the loss has the data loss term and thus
genereated images are more realistic, in the GAN the losses are related to the discriminators ability to 
discriminate real from fake images, in which case it is more possible that the loss would decrease even if
the images has unrealistic features, but this loss forces to generator to generate sharper images in which
the face could be seen more clearly.

"""

# ==============


