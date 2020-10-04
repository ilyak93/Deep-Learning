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


There several main reasons to split corpus into sequences:

1.  Different texts are in different lengths. When using a fixed (relatively small) sequence length
    we can make sure that our architecture is not text-dependent. This is similar to working with batches.
2.  We'd wish that the network will resolve connections between words in many contexts instead of "remembering"
    large amounts of texts. This improves network generalization and provides better quality leaning 
3.  To support parallelism. In this homework we were training in parallel and to do so we count on the
    sequence length to be of fixed sizes.
4.  From a purely technical point of view: it is not realistic to back-propagate through very big net depths.
    We're limited in computing resources and because of that we limit back-propagating into a fixed time length. 
5.  It effectively reduces the dimension of the input to the network, thus
	speeding the training process and allowing for more frequent updates of the parameters.

"""

part1_q2 = r"""
**Your answer:**


As written in q1, the network is able to resolve connections between letters, words, punctuation between different
sequences. Instead of memorizing the network "learns" the rules of the language, meaning it can generalize without
over-fitting the text, and that is because the hidden state is transferred
between batches (it is only zeroed between epochs), thus, when a new batch is generated, it takes
into account the "memory" of the previous batch, thus the network can generate any length new sequences. 

"""

part1_q3 = r"""
**Your answer:**


In order for the network to be able to learn continues sentences we need to maintain sequence order in different
batches, or else every batch would provide a sequence which is not related at all to the last sequence. 
tIn other words, there is a dependency between the batches (as the text is split between
them), and the hidden state is transferred between them. In this case, we don't want to shuffle the batches
so that we don't lose the information of the connection between them.
"""

part1_q4 = r"""
**Your answer:**



1.	When training we wish to learn probability distribution for the next predicted character. Therefore we do not
    need to emphasize small differences yet. When sampling we wish that characters with slightly higher probability
    will have a really high chance of being predicted. Since softmax often provides almost uniformly distribution
    for similar scores we use temperature to avoid that.
    
2.  As T grows the distribution is being more uniform. When "T is infinity" the distribution will be pure uniform,
	in which case the generated text will be almost random.

3.  As T becomes smaller the differences between original scores are being larger. When T is "almost 0" one character
    will have almost probability of 1.0 to be predicted while all of the others will have probability of almost 0 to be
    predicted.  

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

Intuitively the x_sigma2 is the variance around the mean vector in the latent space.
This means that as x_sigma2 gets bigger similar photos from the instance space will be
mapped to more different points in the latent space. In this setting not only a single
point from the latent space will be decoded to a certain photo in the instance space,
but all of the pointer in the "area" around it will be also mapped in to similar photos
in the instance space, so the generated images more resemble the dataset images, 
and we see less variety in poses and backgrounds but a better quality.

The opposite setting, in which s_sigma2 is small, only a small "area" in the latent
space will be mapped to similar photos in the instance space. This, intuitively, 
reduces the mapping "area" between a photo in the instance space to a point in the
latent space. In another view, it causes the generated images to have more randomness, 
thus in these cases we see more variety of poses and backgrounds.

In conclusion, the hyperparameter $\sigma^2$ control the tradeoff between the reconstruction loss and the KLD loss.

"""

part2_q2 = r"""
**Your answer:**


1.  The purpose of the reconstruction loss is to measure how well the encoder-decoder reconstructs
    images. In other words, what is the "difference" between IMG and ENCODE(DECODE(IMG)), while IMG
    is a photo from the instance space.
	in other words, due to the reconstruction loss we are able to maximize the probability that x=xr, 
	i.e increase the probability that the generated image will resemble the images from the dataset.
    
    The KL loss is a measure to define similarity between two probability distributions in the latent
    space. We use the KL loss to measure the difference between our probability distribution in the latent
    space and a normal distribution. The purpose of this loss is to ensure that the distribution in the
    latent space is "close to being normal" (in the sense of KL divergence), or to ensure that two points
    that are close to each other in the latent space will be mapped in to similar points in the instance
    space. 
    
2.  As mentioned above the KL loss enforces the weights of the encoder to generate close to normal distribution
    in the latent space. This is explained above. 
	We can also say that the KL divergence loss purpose is to minimize the 'distance' (in KL manner) between the approximated posterior
	distribution of the latent space and the prior distribution, p(Z), **thus it controls the randomness in the generated
	images**.

3.  We wish to avoid the case in which when we select two close points in the latent space we get totally different
    george bushes. This enables the latent space mapping to the instance space to be smooth

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
more accurate look of the eyes or maybe some clothes details.
These changes are related mainly to the loss differences. While in the VAE, the loss has the data loss term and thus
genereated images are more realistic, in the GAN the losses are related to the discriminators ability to 
discriminate real from fake images, in which case it is more possible that the loss would decrease even if
the images has unrealistic features, but this loss forces to generator to generate sharper images in which
the face could be seen more clearly.

"""

# ==============


