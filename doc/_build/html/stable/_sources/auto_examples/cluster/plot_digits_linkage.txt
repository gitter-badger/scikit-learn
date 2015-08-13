

.. _example_cluster_plot_digits_linkage.py:


=============================================================================
Various Agglomerative Clustering on a 2D embedding of digits
=============================================================================

An illustration of various linkage option for agglomerative clustering on
a 2D embedding of the digits dataset.

The goal of this example is to show intuitively how the metrics behave, and
not to find good clusters for the digits. This is why the example works on a
2D embedding.

What this example shows us is the behavior "rich getting richer" of
agglomerative clustering that tends to create uneven cluster sizes.
This behavior is especially pronounced for the average linkage strategy,
that ends up with a couple of singleton clusters.


**Python source code:** :download:`plot_digits_linkage.py <plot_digits_linkage.py>`

.. literalinclude:: plot_digits_linkage.py
    :lines: 18-
    