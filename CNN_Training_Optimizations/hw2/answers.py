r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.1, 0.05, 0.0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg = 8, 0.02, 0.005, 0.0002, 0.001
	
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 1
    lr = 0.005
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""

1. The results are what we expected to see.
With Dropout=0 the training accuracy was higher than Dropout=0.4 since all of the neurons were able to train and fit the model to the data. But,
this caused the model to be less general and maybe even to over-fit. This can be seen in the test accuracies graph - although the net without
dropout had higher training accuracy than the net with dropout, the test accuracy of the net with dropout was higher.

2. Comparing low dropout to high dropout we can see that the test accuracy of the low dropout net is higher. We conclude that high dropout values
prevents efficient training of the net weights and therefor harms the net's conversion time and ultimately causes under-fitting from one side,
but from another side the right values of dropout can reduce the overfitting.

"""

part2_q2 = r"""

Cross entropy loss function grows as a function of a dissimilarity between the true distribution
of the training set and the predicted distribution.

It is possible that loss will be increased while accuracy is being increased as well. This happens when the model was very confident about
certain sample classes and wasn't certain about other certain classes in which it classified wrong. Cross entropy loss punishes on low
confident level on correct classes and that fact might cause the phenomenon described above.

For example:
Let's assume a binary classification problem with two classes: C1 and C2.
Let there be two samples S1 and S2 and ground truth class C1 for both.

At the beginning S1 was classified C1 with high confidence (p1=0.99999) and S2 was classified C2 with low confidence (p2=0.6).
The accuracy here is 50% while the cross entropy loss is 0.39.
After one training epoch S1 was classified C1 with low confidence (p1=0.6) and S2 was classified C1 with low confidence (p2=0.6).
Now the accuracy is 100% while the cross entropy loss is 0.44.

This examples shows that sometimes the punishment over low confidence might be larger than the reward for correct classification.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
*************************************************************************************************************************************************
In general in our experiments:
****************************************************
Firstly, ir's important to mention that in the run_experiment implementation assignment,
although there are parameters which are given to the function, we used our own, because it is said,
that all other hyperparameters except those we are experimenting on are up to us, 
so for better observations, we didn't limit the batch size by the default 100,
thus we got a full batch size of 391 batches each 128 samples, which mean that each epoch we observe
the whole dataset with a shuffle set to True in the DataLoader. That mean that each run we can get different results,
which depend some on the randomness of the shuffling. Also for better observation of the loss curve and accurancy curve,
we set early stopping to 5, of course we could observe for the shorter (in terms of depth) network some overfit,
but seting early stopping to lbe less or enabling checkpoints we always culd get the previous better minimum of the optimization,
and by setting it to 5 we could see better the behavior of the curve. We also set the batch test size w.r.t the training batch size
and the relation between the whole training dataset size and the test dataset size, being 5 to 1.
One conclusion from not limiting the batch size, is that observing the whole dataset each epoch gave better convergence at more deepest 
network, as you can see from all of the graphs in all parts this assignment, and that improved the accurancy with 5-7%, as expected,
as not seeing the whole dataset in each epoch could mean, we're seeing less data, means less good convergence. 
**************************************************************************************************************************************************

1.
	Now, in those good conditions for making observations, we can emphasize the following conclusions:
****************************************************
	a. There was some overfitting, as the train accurancy is bigger, then test accurancy, which is partially caused by enlarging the early stop criteria,
	as was allready mentioned, that for the less deep network we could stop by early stopping earlier, but for getting better results for L=8,
	it was essential, as the curve climbs very slow.
****************************************************
	b. There is pretty good convergence with all the parameters to the area of 70% accurancy with K=32
****************************************************
	c. As expected, deeper network got better accurancy results and convergence: L=2 about 68%, L=4 about 70%, L=8 about 73-74%,
	but as we can see for L=8 the network is much stable in the area of the minimal loss, as it continued to run and get better accurancy.
****************************************************
	d. It is aso observable, that we see the vanishing gradient issue in the L=8 depth, because the curve become much more moderate.
****************************************************

Same conclusions with K=64, but the accurancies in all three depths are better about 2% in each. It isn't necessarily expected, but as we can tell
from the results, it means that we can extract more useable and efficient features, it is probably data driven hyperparameter, and in our case, 
it seems to be that more filters is better. Although in our case, our data set is big enough so, it is probably expected that bigger number of filters
will give us better feature extraction.

As expected in generally and following the above conclusions,
 the best result were obtained fot the deepest trainable network with L=8, with about 76% aacurancy on test set. 

****************************************************
2. For L=16 wasn't trainable at all. The reason for the in my opinion, that the output extent became too small for the FC to make any sufficient gradient updates, as much as
classifications, because it is too hard to give the right label of a number of image 32*32 ending with a representation of 2*2 of some amount of filters, 
i.e it a bad and weak expression of the features of the image.
As I saw from debbuging it, and as we can calculate from the formula in the tutorial, with stride=2 and pool_every=4, as were taken in our algorithm, maxPool layer
decreasing the extent size by factor 2 each 2 layer which means, that the output extent is 32 / 2^4 = 2, and if we pool with greater frequency, as was taken for L=2,4,8,
it will vanish at all for the minimum allowed 1 or isn't runnable at all, so it too less information for classify only 2 by 2 extent with depth 32, i.e 2*2*32.
****************************************************  
The first way to fix this is to decrease the kernel size. This will cause less convolutions that will not decrease the data dimension so
drastically.
The second way to either pad (which will prevent drastic decrease of the dimensions) or pool less frequently, 
which was used by us: instead pooling each 2, where the extent size is becoming 0, we pooled each 4 at least to see the "too small output extent" of the feature extractor,
as mentioned. 
****************************************************
"""

part3_q2 = r"""


Firstly, as was observed in the 1.1 and expected to appear here, deeper length, i.e across the enlargmenet of L parameter for all of the K values,
gave better accurancy. 

For the L=4,L=8 as the conclusion about the filter extent size observed for the comparasion between K=32 and K=64, wih enough data, as we have,
 we can extract more useable and efficient features, thus more filters is better.

Explaining the L2 curves:
It holds eather for the L=2 depth, except for K=256, and that is for my opinion because we need more depth, 
so the model could sufficiently use it's expression power of 256 filters. 
L=2 curves don't contradict this conclusion, for L=2 there isn't enough depth to get the influence of
the filters number, because with more depth the features became more expressive and the potential of the data expressed. 
That's why we see that for L=2 we got the best results, i.e for suffiency of big K (filter extent size) we need good enough depth.

As expected in generally and following the above conclusions the best result were obtained fot the deepest network fot the second biggest K value(128),
 with about 78% aacurancy on test set. 

"""

part3_q3 = r"""

The results are relationally better, or to be more exact the results of L2, L3 depths are at least good and in many cases better, then
same results of constant same K values or bigger L values in the previous experiments, which can tell us that tis is a good practice, 
to set filters with enlarging size. That way we achive good results with less deep networks. 

Accordingly, the L1 also was better then any of the L2 with different constant K, and is comparable for L4 with K=32.

As expected in generally and following the above conclusions the best result were obtained fot the deepest trainable network fot the second biggest L value(3),
 with about 78% aacurancy on test set.
"""

part3_q4 = r"""

Here the model is capable to train with very large L values! Furthermore its accuracy is better than the previous experiments with same filters number K=32,
by about 2-3%.
This is duo to the fact that we're using a residual network. The network helps us preserve additional information from previous
layers. Therefore, even when the input dimension for the fully connected layer is small we still get good results, even better
than before. The data from the previous layers, which are in a bigger dimension, is projected to the lower dimension with relatively
big, and therefore significant values.

For K=64,128,256 the results for L=2 was the same accurency and comparing L=4 and L=8 to L=2 and L=3 in 1.3, in resnet the results were a bit less acurate,
then in 1.3. 

The best results are for L=2 with the list of K values (same was as 1.3) with about 78% aacurancy on test set. 

"""

part3_q5 = r"""

1. First in this model we have enabled batch normalization and increased the dropout value to be 0.4 for better generalization as seen previously
in Part 2. Additionally we useв residual blocks with max pooling blocks with zero-padding and bigger kernel size of max pooling.
This was done considering the goal of the output dimension of the feature extractor to be as big as possible as a conclusion from all the experiments
in Part 3. That gave us better results comparing to its similar architecture of 64-128-256 with residentual blocks 1.4 and similiar results to the 1.3 results
with cnn with 64-128-256 filters. Of course as expected much better results then 1.1 and 1.2 experiments.   

We experimneted with the max pooling hyperparameters and generally gound that bigger kernel (3,4) gave better accurancy results, 
and to make the network robust, we added padding as was mentioned. We also tried different pool parameter adjustments, to get the best results.

Trying enlarge the FC size or add another hidden layer didn't influence the results at all. 

We also consider changing the kernel sizes but this works against our goal of increasing the output dimension. Also we're using Adam
optimizer which is known to achieve great results. It balances between RMS-Promp (which gives larger learning rate for rarely updated parameters)
and Momentum (which increases learning rate for a repetitive direction).

2. The best result we got is for 3-6 layers, the test accuracy was about 80%. 
Our result were similar to residual network/ cnn with different filter sizes,
in terms of test accuracy but were much more robust to increasing number of layers. 
As seen previously, there were some not very big gaps between models with different layer numbers, about few %. 

Compared to CNN we've got better test accuracy with bigger amount of epochs. Additionally needed only 3 layers to outperform CNNs with more layers.
Compared to Residual nets our results are better either.

By our curiosity was made another experiment with bigger filter sizes with our net, as was done in previous experiments,
we used 64-128-256 size filters and that gave us even better results of all we've decribed untill now.
The best result was with L3 depth, as it converge very fast, with about 84% test accurancy,
where other depths as 6 and 9 gave about 80-82 test accurancies.   

"""
# ==============
