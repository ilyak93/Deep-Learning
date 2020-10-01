r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(batch_size=32,
              gamma=0.99,
              beta=0.5,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    hp["batch_size"] = 32
    hp["gamma"] = 0.99
    hp["beta"] = 0.5
    hp["learn_rate"] = 3 * 1e-3
    hp["eps"] = 1e-8
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(batch_size=32,
              gamma=0.99,
              beta=1.,
              delta=1.,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    hp["batch_size"] = 16
    hp["gamma"] = 0.99
    hp["beta"] = 0.9
    hp["delta"] = 0.35
    hp["learn_rate"] = 3e-3
    hp["eps"] = 1e-8
    # ========================
    return hp



part1_q1 = r"""

Let's imagine two situations:

1) Trajectory A receives +10 reward and trajectory B receives -10 reward.
2) Trajectory A receives +10 reward and trajectory B receives +1 reward.

Without any baseline in situation (1) the probability of choosing A will increase
while choosing B will decrease.

While in situation (2), probabilities for both trajectories will increase.

For humans it is obvious that in both situations the probability of B should decrease while
the probability A should increase. That's because the reward raw value itself does not mean
anything on its own but only in comparison to the other rewards.

When no baseline exists the actual baseline is 0 which is random and does not mean anything.

"""


part1_q2 = r"""
**Your answer:**


When we used the regular baseline method we calculated it by taking an average over
the q-values. Now to calculate the new baseline we use a neural network with a loss
function which is the MSE between the state value and the q-value. The state value
(v_pi(s)) is the expectation of the q-values after taking each action.

Therefore the approximation is valid because it is based on the same values we used
to calculate the old baseline in the first place. 

"""


part1_q3 = r"""
**Your answer:**


1)
In the first experiment we can see several results:
a. The loss_p of baseline based methods (bpg, cpg) close to zero
b. Eventually all experiments achieved positive best reward 
   (the hyperparameters which are good to the baseline methods aren't same good to the epg and vpg methods, 
   but if we tweak them, we can get this result).
c. The baseline based methods achieves good rewards faster than other methods

We think we can conclude from these results that:
a. Baseline based methods doesn't calculate absolute loss, it calculates the loss in
comparison to the "average" loss. Therefor we expect it to be much closer to zero
b. This means that the nets avtually succeeded in learning how to play this game better
c. It seems that comparing a trajectory to its "rivals" and choosing the trajectory
accordingly is really getting better learning results as expected

2)
The loss_p of aac ascends very quickly, which means that the adventage minimizing boosts the learning to get better results.
The second learning process of estimating v, contributes segnificantly to the learning rate of the acc.
Also we can see that ,over time, aac passes bpg. That is probably because approximating each v individually over time eliminates more learning mistakes than approximating the whole group to a given average. when approximating a group to the average we are more exposed for big variance and therefore for more mistakes. 

"""
