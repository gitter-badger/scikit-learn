

.. _example_applications_plot_model_complexity_influence.py:


==========================
Model Complexity Influence
==========================

Demonstrate how model complexity influences both prediction accuracy and
computational performance.

The dataset is the Boston Housing dataset (resp. 20 Newsgroups) for
regression (resp. classification).

For each class of models we make the model complexity vary through the choice
of relevant model parameters and measure the influence on both computational
performance (latency) and predictive power (MSE or Hamming Loss).


**Python source code:** :download:`plot_model_complexity_influence.py <plot_model_complexity_influence.py>`

.. literalinclude:: plot_model_complexity_influence.py
    :lines: 16-
    