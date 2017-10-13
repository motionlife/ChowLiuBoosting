# ChowLiuBoosting
Use boosting method in graphical models to improve the classification accuracy rate
### Weaker learner
Use Chow-liu algorithm to learn the generative tree distribution of weighted data in each round. Consider our classification task, we only need to store 
the marginal distribution of the label and the pairwise marginal distribution with its neighbours in the chow-liu tree.
### Boosting
For multi-class boosting task, we choose SAMME and SAMME.R algorithm, the later is a soft confidence-rated compared the a hard score rating.
