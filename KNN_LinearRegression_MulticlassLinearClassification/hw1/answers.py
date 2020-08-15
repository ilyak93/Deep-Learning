r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""

The best value of K is somewhere in the middle and to find the best one we need to perform tuning by testing cross-validation.

Increasing the value of K does not guarentee better results. This can be seen from our results.

In the most extereme case where K equals to the whole database size we will always predict the most common class.

Where K is 1 we might mis-predict if the nearest neighbour is an outlier.

Therefore we cannot avoid testing and cross-validations to find the optimal K value.

"""

part2_q2 = r"""

1. Choosing the best model by train-set accuracy is problematic since every prediction is made on a sample already seen and trained by.

Therefore we do not check the generalization of our model by doing so.

When talking about KNN classifier - we will always choose K=1 since it will lead to 100% accuracy since every sample is already seen and memorized.

2. Choosing the best model by test-set accuracy can lead to overfitting and increase bias due to non-representitive data.

We might adjust our model only for a sub-population that exists in the test.

By performing k-fold CV we reduce the probability to overfitting and possibly increase the ability for better generalization.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


If we used SVM loss without regularization, choosing delta would effect the weights magnitude. For example, very large delta will cause large loss which will enforce the weights

to be the same values but in larger orders of magnitudes. On the other hand, if we use regularization large weights magnitudes will be "punished" but with big loss value by the

regularization term. There we can see that the two (delta and lambda) create the same effect on weights magnitudes. Therefore, we can choose an arbitrary value for delta and then

control weights magnitudes only by twitching lambda values.

"""

part3_q2 = r"""

1. what the linear model is learning is which pixel is important (and how important) for every class. For example, for the digit 2's most important pixels will probably be in the

relatively bottom and upper part of the image. The straight lines visualized in this excersise (visualizing the weights) can be intuitively explained as the general digit, meaning

if we draw a digit and then move it along the screen (since it can be drawned anywhere on the picture) it will be drawn as straight lines.

Digits 5 and 6 weight maps looks very similar. This make sense by the expanation above. Therefore we can see a lot of classification errors for those digits

(classifying 6 as 5 and vice versa).

2. KNN compares multi-dimentional representations of tagged samples to a new sample by comparing to the K nearest neighbors in order to classify the new sample.

In our interpolation we don't compare our new sample to constant number of samples, we compare each sample to the general representation of each digit which was

created in the learning phase. This fact helps our model to both classify instantly (whithout needing to compare to a lot of images) and to gain generalization.

"""

part3_q3 = r"""

1. We think that according to the training-set graph our learning rate was good :D

If it was too high then we had harder time to converge and could probably see spikes suggesting skipping the minimum from different directions.

If it was too low we would converge very slowly, unlike the graph which we converge very fast at the beginning and afterwards we keep converging slowly.

2. We think that by looking at the accuracy graph we can conclude that our model is slightly overfitted to the training data. We think that because in the

beginning the model's accuracy improved significantly due to "real" learning and after it reaches a certain point the leaning curve is a lot smaller. From

this point on the validation accuracy is not really improving while training accuracy keeps growing. We would recommend stopping the learning a bit before.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""


The residual plot is great for visualizing the generalization ability of the trained model. The ideal pattern to see (the one which indicates of a good generalization)

is a pattern where most of the data is between the dotted lines but not all of it: this indicates about good estimation (since most of the data is classified correctly)

and good generalization (since not every noisy data point causes overfitting). Also, if the train and test results are fairly similar it indicates of great generalization.

In our example the model chosen by top 5 features training has good generalization abilities because it fits most of the data as described above. Furthermore, the

benefit from using higher degree is, in this example, pretty low. degree of 3 led to bad results while degree of 2 improved the MSE loss slightly.

"""

part4_q2 = r"""



1. Using logspace enables to explore a large scale of regularization values in order to test the trade-off between generalization and estimation error. Big lambda values

will favour generalization while smaller lambdas will favour estimation. It is hard to guess the regularization value and therefore using logspace will help test more

value magnitudes.

2. The model was fitted: K-fold * degrees range * lambdas range, which in our case is: 3*4*20=240.

"""

# ==============
