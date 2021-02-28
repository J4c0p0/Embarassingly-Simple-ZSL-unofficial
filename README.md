# Embarassingly-Simple-ZSL-unofficial

This is an unofficial, third-party, implementation of the ICML 2015 paper entitled "An embarrassingly simple approach to zero-shot learning" (<a href="http://proceedings.mlr.press/v37/romera-paredes15.html">ESZSL</a>) by Bernardino Romera-Paredes and Philip Torr, on <a href="https://cvml.ist.ac.at/AwA2/">Aninamls With Attributes</a> dataset.

Please note that this repository was created after an unoffical demo tutorial session that I provided in front of my lab: the code is not meant to be performance-oriented, since it is didactical in scope. In fact, for an accurate historical reconstruction, I considered <a href="https://github.com/Elyorcv/SAE">GoogleNet features and splits</a> by the paper [Kodirov et al., Semantic Autoencoders for Zero-Shot Learning, CVPR 2017] which should match the ones used by ESZSL. I am thus not considering the recommended "Proposed Splits" by [Xian et al. Zero Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly, TPAMI 18] which has to be preferred in practice, but were not simply available at the time ESZSL was published. We still borrow this reposityory to use the <a href="https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly">list of attributes</a>, tabulated suing Osherson's <a href="https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1502_3">defaul probabilty</a> scores, provided therein.

## Prerequisites
`````
MATLAB
