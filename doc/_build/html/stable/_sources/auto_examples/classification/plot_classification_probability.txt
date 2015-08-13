

.. _example_classification_plot_classification_probability.py:


===============================
Plot classification probability
===============================

Plot the classification probability for different classifiers. We use a 3
class dataset, and we classify it with a Support Vector classifier, L1
and L2 penalized logistic regression with either a One-Vs-Rest or multinomial
setting.

The logistic regression is not a multiclass classifier out of the box. As
a result it can identify only the first class.


**Python source code:** :download:`plot_classification_probability.py <plot_classification_probability.py>`

.. literalinclude:: plot_classification_probability.py
    :lines: 14-
    