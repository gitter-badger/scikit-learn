import warnings
import unittest
import sys
import numpy as np
from scipy import sparse as sp

from nose.tools import assert_raises
from sklearn.utils.testing import (
    _assert_less,
    _assert_greater,
    assert_less_equal,
    assert_greater_equal,
    assert_warns,
    assert_no_warnings,
    assert_equal,
    set_random_state,
    assert_raise_message,
    assert_safe_sparse_allclose,
    assert_same_model,
    assert_not_same_model)
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.qda import QDA
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans

try:
    from nose.tools import assert_less

    def test_assert_less():
        # Check that the nose implementation of assert_less gives the
        # same thing as the scikit's
        assert_less(0, 1)
        _assert_less(0, 1)
        assert_raises(AssertionError, assert_less, 1, 0)
        assert_raises(AssertionError, _assert_less, 1, 0)

except ImportError:
    pass

try:
    from nose.tools import assert_greater

    def test_assert_greater():
        # Check that the nose implementation of assert_less gives the
        # same thing as the scikit's
        assert_greater(1, 0)
        _assert_greater(1, 0)
        assert_raises(AssertionError, assert_greater, 0, 1)
        assert_raises(AssertionError, _assert_greater, 0, 1)

except ImportError:
    pass


def test_assert_less_equal():
    assert_less_equal(0, 1)
    assert_less_equal(1, 1)
    assert_raises(AssertionError, assert_less_equal, 1, 0)


def test_assert_greater_equal():
    assert_greater_equal(1, 0)
    assert_greater_equal(1, 1)
    assert_raises(AssertionError, assert_greater_equal, 0, 1)


def test_set_random_state():
    lda = LinearDiscriminantAnalysis()
    tree = DecisionTreeClassifier()
    # Linear Discriminant Analysis doesn't have random state: smoke test
    set_random_state(lda, 3)
    set_random_state(tree, 3)
    assert_equal(tree.random_state, 3)


def test_assert_raise_message():
    def _raise_ValueError(message):
        raise ValueError(message)

    def _no_raise():
        pass

    assert_raise_message(ValueError, "test",
                         _raise_ValueError, "test")

    assert_raises(AssertionError,
                  assert_raise_message, ValueError, "something else",
                  _raise_ValueError, "test")

    assert_raises(ValueError,
                  assert_raise_message, TypeError, "something else",
                  _raise_ValueError, "test")

    assert_raises(AssertionError,
                  assert_raise_message, ValueError, "test",
                  _no_raise)

    # multiple exceptions in a tuple
    assert_raises(AssertionError,
                  assert_raise_message, (ValueError, AttributeError),
                  "test", _no_raise)


def test_assert_safe_sparse_allclose():
    x = 1e-3
    y = 1e-9
    assert_safe_sparse_allclose(x, y, atol=1)
    assert_raises(AssertionError, assert_safe_sparse_allclose, x, y)

    a = sp.csc_matrix(np.array([x, y, x, y]))
    b = sp.coo_matrix(np.array([x, y, x, x]))
    assert_safe_sparse_allclose(a, b, atol=1)
    assert_raises(AssertionError, assert_safe_sparse_allclose, a, b)

    b[-1] = y * (1 + 1e-8)
    assert_safe_sparse_allclose(a, b)
    assert_raises(AssertionError, assert_safe_sparse_allclose, a, b,
                  rtol=1e-9)

    assert_safe_sparse_allclose([np.array([(6, 6)]),], [np.array([(10, 10)]),],
                                rtol=0.5)
    assert_raises(AssertionError, assert_safe_sparse_allclose,
                  [np.array([(6, 6)]),], [np.array([(10, 10)]),], rtol=0.5)

    a = sp.csr_matrix(np.array([np.iinfo(np.int_).min], dtype=np.int_))
    # Should not raise:
    assert_allclose(a, a)

    # Test nested lists of scalars
    assert_safe_sparse_allclose([(['a', 'bcd'], ['a'])],
                                [(['a', 'bcd'], ['a'])])
    assert_raises(AssertionError, assert_safe_sparse_allclose,
                  [(['a', 'bcd'], ['a'])], [(['a', 'bcd'], ['a', 'a'])])
    assert_raises(AssertionError, assert_safe_sparse_allclose,
                  [(['a', 'bcd'], ['a'])], [(['a', 'bcd'], ['b'])])


    # Test the string comparison
    assert_safe_sparse_allclose('a', 'a')
    assert_safe_sparse_allclose('abcdl', 'abcdl')
    assert_raises(AssertionError, assert_safe_sparse_allclose, 'a', 'b')
    assert_raises(AssertionError, assert_safe_sparse_allclose, 'aa', 'b')

    # Test numeric comparisons
    assert_safe_sparse_allclose(6, np.float64(6))
    assert_safe_sparse_allclose(6, 6.0)
    assert_safe_sparse_allclose(7, 7.0)
    assert_safe_sparse_allclose(5, np.int32(5))

    # Make sure you don't get infinite recursion with empty nested lists
    x = []
    x.append(x)
    assert_safe_sparse_allclose(x, x)


def test_assert_same_not_same_model():
    X1, y1 = make_blobs(n_samples=200, n_features=5, center_box=(-200, -150),
                        centers=2, random_state=0)
    X2, y2 = make_blobs(n_samples=100, n_features=5, center_box=(-1, 1),
                        centers=3, random_state=1)
    X3, y3 = make_blobs(n_samples=50, n_features=5, center_box=(-100, -50),
                        centers=4, random_state=2)

    # Checking both non-transductive and transductive algorithms
    # By testing for transductive algorithms we also eventually test
    # the assert_fitted_attributes_equal helper.
    for Estimator in (LinearSVC, KMeans):
        assert_same_model(X3, Estimator(random_state=0).fit(X1, y1),
                          Estimator(random_state=0).fit(X1, y1))
        assert_raises(AssertionError, assert_not_same_model, X3,
                      Estimator(random_state=0).fit(X1, y1),
                      Estimator(random_state=0).fit(X1, y1))
        assert_raises(AssertionError, assert_same_model, X3,
                      Estimator(random_state=0).fit(X1, y1),
                      Estimator(random_state=0).fit(X2, y2))
        assert_not_same_model(X3, Estimator(random_state=0).fit(X1, y1),
                              Estimator(random_state=0).fit(X2, y2))


def test_qda_same_model():
    # NRT to make sure the rotations_ attribute is correctly compared
    X = np.array([[0, 0], [-2, -2], [-2, -1], [-1, -1], [-1, -2],
                  [1, 3], [1, 2], [2, 1], [2, 2]])
    y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])
    X1 = np.array([[-3, -1], [-2, 0], [-1, 0], [-11, 0], [0, 0], [1, 0],
                   [1, 5], [2, 0], [3, 4]])
    y1 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2])
    X2 = np.array([[-1, -3], [0, -2], [0, -1], [0, -5], [0, 0], [10, 1],
                  [0, 11], [0, 22], [0, 33]])

    clf1 = QDA().fit(X, y)
    clf2 = QDA().fit(X, y)
    assert_same_model(X1, clf1, clf2)

    clf3 = QDA().fit(X1, y1)
    assert_not_same_model(X2, clf1, clf3)


# This class is inspired from numpy 1.7 with an alteration to check
# the reset warning filters after calls to assert_warns.
# This assert_warns behavior is specific to scikit-learn because
# `clean_warning_registry()` is called internally by assert_warns
# and clears all previous filters.
class TestWarns(unittest.TestCase):
    def test_warn(self):
        def f():
            warnings.warn("yo")
            return 3

        # Test that assert_warns is not impacted by externally set
        # filters and is reset internally.
        # This is because `clean_warning_registry()` is called internally by
        # assert_warns and clears all previous filters.
        warnings.simplefilter("ignore", UserWarning)
        assert_equal(assert_warns(UserWarning, f), 3)

        # Test that the warning registry is empty after assert_warns
        assert_equal(sys.modules['warnings'].filters, [])

        assert_raises(AssertionError, assert_no_warnings, f)
        assert_equal(assert_no_warnings(lambda x: x, 1), 1)

    def test_warn_wrong_warning(self):
        def f():
            warnings.warn("yo", DeprecationWarning)

        failed = False
        filters = sys.modules['warnings'].filters[:]
        try:
            try:
                # Should raise an AssertionError
                assert_warns(UserWarning, f)
                failed = True
            except AssertionError:
                pass
        finally:
            sys.modules['warnings'].filters = filters

        if failed:
            raise AssertionError("wrong warning caught by assert_warn")
