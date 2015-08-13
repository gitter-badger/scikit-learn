

.. _example_cross_decomposition_plot_compare_cross_decomposition.py:


===================================
Compare cross decomposition methods
===================================

Simple usage of various cross decomposition algorithms:
- PLSCanonical
- PLSRegression, with multivariate response, a.k.a. PLS2
- PLSRegression, with univariate response, a.k.a. PLS1
- CCA

Given 2 multivariate covarying two-dimensional datasets, X, and Y,
PLS extracts the 'directions of covariance', i.e. the components of each
datasets that explain the most shared variance between both datasets.
This is apparent on the **scatterplot matrix** display: components 1 in
dataset X and dataset Y are maximally correlated (points lie around the
first diagonal). This is also true for components 2 in both dataset,
however, the correlation across datasets for different components is
weak: the point cloud is very spherical.


**Python source code:** :download:`plot_compare_cross_decomposition.py <plot_compare_cross_decomposition.py>`

.. literalinclude:: plot_compare_cross_decomposition.py
    :lines: 21-
    