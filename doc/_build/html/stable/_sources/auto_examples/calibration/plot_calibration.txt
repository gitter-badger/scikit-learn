

.. _example_calibration_plot_calibration.py:


======================================
Probability calibration of classifiers
======================================

When performing classification you often want to predict not only
the class label, but also the associated probability. This probability
gives you some kind of confidence on the prediction. However, not all
classifiers provide well-calibrated probabilities, some being over-confident
while others being under-confident. Thus, a separate calibration of predicted
probabilities is often desirable as a postprocessing. This example illustrates
two different methods for this calibration and evaluates the quality of the
returned probabilities using Brier's score
(see http://en.wikipedia.org/wiki/Brier_score).

Compared are the estimated probability using a Gaussian naive Bayes classifier
without calibration, with a sigmoid calibration, and with a non-parametric
isotonic calibration. One can observe that only the non-parametric model is able
to provide a probability calibration that returns probabilities close to the
expected 0.5 for most of the samples belonging to the middle cluster with
heterogeneous labels. This results in a significantly improved Brier score.


**Python source code:** :download:`plot_calibration.py <plot_calibration.py>`

.. literalinclude:: plot_calibration.py
    :lines: 23-
    