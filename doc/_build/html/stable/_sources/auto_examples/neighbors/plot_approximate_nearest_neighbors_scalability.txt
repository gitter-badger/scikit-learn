

.. _example_neighbors_plot_approximate_nearest_neighbors_scalability.py:


============================================
Scalability of Approximate Nearest Neighbors
============================================

This example studies the scalability profile of approximate 10-neighbors
queries using the LSHForest with ``n_estimators=20`` and ``n_candidates=200``
when varying the number of samples in the dataset.

The first plot demonstrates the relationship between query time and index size
of LSHForest. Query time is compared with the brute force method in exact
nearest neighbor search for the same index sizes. The brute force queries have a
very predictable linear scalability with the index (full scan). LSHForest index
have sub-linear scalability profile but can be slower for small datasets.

The second plot shows the speedup when using approximate queries vs brute force
exact queries. The speedup tends to increase with the dataset size but should
reach a plateau typically when doing queries on datasets with millions of
samples and a few hundreds of dimensions. Higher dimensional datasets tends to
benefit more from LSHForest indexing.

The break even point (speedup = 1) depends on the dimensionality and structure
of the indexed data and the parameters of the LSHForest index.

The precision of approximate queries should decrease slowly with the dataset
size. The speed of the decrease depends mostly on the LSHForest parameters and
the dimensionality of the data.



**Python source code:** :download:`plot_approximate_nearest_neighbors_scalability.py <plot_approximate_nearest_neighbors_scalability.py>`

.. literalinclude:: plot_approximate_nearest_neighbors_scalability.py
    :lines: 30-
    