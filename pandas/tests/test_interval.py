import numpy as np
import unittest

from pandas.core.interval import Interval, IntervalIndex
from pandas.core.index import Index

import pandas.util.testing as tm
import pandas as pd


class TestInterval(tm.TestCase):
    def setUp(self):
        self.interval = Interval(0, 1)

    def test_properties(self):
        self.assertEqual(self.interval.closed, 'right')
        self.assertEqual(self.interval.left, 0)
        self.assertEqual(self.interval.right, 1)
        self.assertEqual(self.interval.mid, 0.5)

    def test_repr(self):
        self.assertEqual(repr(self.interval),
                         "Interval(0, 1, closed='right')")
        self.assertEqual(str(self.interval), "(0, 1]")

        interval_left = Interval(0, 1, closed='left')
        self.assertEqual(repr(interval_left),
                         "Interval(0, 1, closed='left')")
        self.assertEqual(str(interval_left), "[0, 1)")

    def test_contains(self):
        self.assertIn(0.5, self.interval)
        self.assertIn(1, self.interval)
        self.assertNotIn(0, self.interval)
        self.assertRaises(TypeError, lambda: self.interval in self.interval)

        interval = Interval(0, 1, closed='both')
        self.assertIn(0, interval)
        self.assertIn(1, interval)

        interval = Interval(0, 1, closed='neither')
        self.assertNotIn(0, interval)
        self.assertIn(0.5, interval)
        self.assertNotIn(1, interval)

    def test_equal(self):
        self.assertEqual(Interval(0, 1), Interval(0, 1, closed='right'))
        self.assertNotEqual(Interval(0, 1), Interval(0, 1, closed='left'))

    def test_comparison(self):
        self.assertTrue(Interval(0, 1) < 2)
        self.assertTrue(Interval(0, 1) > 0)
        self.assertTrue(Interval(0, 1) < Interval(1, 2))
        self.assertTrue(Interval(0, 1) <= Interval(0, 1))

        self.assertFalse(Interval(0, 1) <= Interval(0.5, 1.5))
        self.assertFalse(Interval(0, 1) < 1)
        self.assertFalse(Interval(0, 1, closed='left') >= 0)

    def test_hash(self):
        # should not raise
        hash(self.interval)

    @unittest.skip('no arithmetic yet')
    def test_math(self):
        expected = Interval(1, 2)
        actual = self.interval + 1
        self.assertEqual(expected, actual)


class TestIntervalIndex(tm.TestCase):
    def setUp(self):
        self.index = IntervalIndex([0, 1], [1, 2])

    def test_constructors(self):
        expected = self.index
        actual = IntervalIndex.from_breaks(np.arange(3), closed='right')
        self.assertTrue(expected.equals(actual))

        alternate = IntervalIndex.from_breaks(np.arange(3), closed='left')
        self.assertFalse(expected.equals(alternate))

        actual = IntervalIndex.from_intervals([Interval(0, 1), Interval(1, 2)])
        self.assertTrue(expected.equals(actual))

        self.assertRaises(ValueError, IntervalIndex, [0], [1], closed='invalid')

        # TODO: fix all these commented out tests (here and below)

        # intervals = [Interval(0, 1), Interval(1, 2, closed='left')]
        # with self.assertRaises(ValueError):
        #     IntervalIndex.from_intervals(intervals)

        # actual = Index([Interval(0, 1), Interval(1, 2)])
        # self.assertTrue(expected.equals(actual))

        # no point in nesting periods in an IntervalIndex
        # self.assertRaises(ValueError, IntervalIndex.from_breaks,
        #                   pd.period_range('2000-01-01', periods=3))

    def test_properties(self):
        self.assertEqual(len(self.index), 2)
        self.assertEqual(self.index.size, 2)

        self.assert_numpy_array_equal(self.index.left, [0, 1])
        self.assertIsInstance(self.index.left, pd.Index)

        self.assert_numpy_array_equal(self.index.right, [1, 2])
        self.assertIsInstance(self.index.right, pd.Index)

        self.assert_numpy_array_equal(self.index.mid, [0.5, 1.5])
        self.assertIsInstance(self.index.mid, pd.Index)

        self.assertEqual(self.index.closed, 'right')

        expected = np.array([Interval(0, 1), Interval(1, 2)], dtype=object)
        self.assert_numpy_array_equal(np.asarray(self.index), expected)
        self.assert_numpy_array_equal(self.index.values, expected)

    def test_delete(self):
        expected = IntervalIndex.from_breaks([1, 2])
        actual = self.index.delete(0)
        self.assertTrue(expected.equals(actual))

    def test_insert(self):
        expected = IntervalIndex.from_breaks(range(4))
        actual = self.index.insert(2, Interval(2, 3))
        self.assertTrue(expected.equals(actual))

        self.assertRaises(ValueError, self.index.insert, 0, 1)
        self.assertRaises(ValueError, self.index.insert, 0,
                          Interval(2, 3, closed='left'))

    def test_take(self):
        actual = self.index.take([0, 1])
        self.assertTrue(self.index.equals(actual))

        expected = IntervalIndex([0, 0, 1], [1, 1, 2])
        actual = self.index.take([0, 0, 1])
        self.assertTrue(expected.equals(actual))

    def test_monotonic_and_unique(self):
        self.assertTrue(self.index.is_monotonic)
        self.assertTrue(self.index.is_unique)

        idx = IntervalIndex.from_tuples([(0, 1), (2, 3)])
        self.assertTrue(idx.is_monotonic)

        idx = IntervalIndex.from_tuples([(0, 1), (1, 2)], closed='left')
        self.assertTrue(idx.is_monotonic)

        idx = IntervalIndex.from_tuples([(0, 1), (0.5, 1.5)])
        self.assertFalse(idx.is_monotonic)
        self.assertTrue(idx.is_unique)

        idx = IntervalIndex.from_tuples([(0, 2), (1, 3)])
        self.assertFalse(idx.is_monotonic)

        idx = IntervalIndex.from_tuples([(0, 1), (1, 2)], closed='both')
        self.assertFalse(idx.is_monotonic)

        idx = IntervalIndex.from_tuples([(0, 2), (0, 2)])
        self.assertFalse(idx.is_unique)
        # self.assertTrue(idx.is_monotonic)

    def test_repr(self):
        expected = ("<class 'pandas.core.interval.IntervalIndex'>\n"
                    "(0, 1]\n(1, 2]\nLength: 2, Closed: 'right', Freq: None")
        IntervalIndex((0, 1), (1, 2), closed='right')
        self.assertEqual(repr(self.index), expected)

    def test_get_loc_value(self):
        self.assertRaises(KeyError, self.index.get_loc, 0)
        self.assertEqual(self.index.get_loc(0.5), 0)
        self.assertEqual(self.index.get_loc(1), 0)
        self.assertEqual(self.index.get_loc(1.5), 1)
        self.assertEqual(self.index.get_loc(2), 1)
        self.assertRaises(KeyError, self.index.get_loc, -1)
        self.assertRaises(KeyError, self.index.get_loc, 3)

        idx = IntervalIndex.from_tuples([(0, 2), (1, 3)])
        self.assertEqual(idx.get_loc(0.5), 0)
        self.assertEqual(idx.get_loc(1), 0)
        self.assertEqual(idx.get_loc(1.5), slice(0, 2))
        self.assertEqual(idx.get_loc(2), slice(0, 2))
        self.assertEqual(idx.get_loc(3), 1)
        self.assertRaises(KeyError, idx.get_loc, 3.5)

        idx = IntervalIndex([0, 2], [1, 3])
        self.assertRaises(KeyError, idx.get_loc, 1.5)

    def test_slice_locs(self):
        self.assertEqual(self.index.slice_locs(), (0, 2))
        self.assertEqual(self.index.slice_locs(0, 1), (0, 1))
        self.assertEqual(self.index.slice_locs(0, 2), (0, 2))
        self.assertEqual(self.index.slice_locs(0, 0.5), (0, 1))
        self.assertEqual(self.index.slice_locs(start=1), (0, 2))
        self.assertEqual(self.index.slice_locs(start=1.2), (1, 2))
        self.assertEqual(self.index.slice_locs(end=1), (0, 1))

    def test_get_loc_interval(self):
        self.assertEqual(self.index.get_loc(Interval(0, 1)), 0)
        self.assertEqual(self.index.get_loc(Interval(0, 0.5)), 0)
        self.assertEqual(self.index.get_loc(Interval(0, 1, 'left')), 0)
        self.assertRaises(KeyError, self.index.get_loc, Interval(2, 3))
        self.assertRaises(KeyError, self.index.get_loc, Interval(-1, 0, 'left'))

    def test_get_indexer(self):
        actual = self.index.get_indexer([-1, 0, 0.5, 1, 1.5, 2, 3])
        expected = [-1, -1, 0, 0, 1, 1, -1]
        self.assert_numpy_array_equal(actual, expected)

        actual = self.index.get_indexer(self.index)
        expected = [0, 1]
        self.assert_numpy_array_equal(actual, expected)

        index = IntervalIndex.from_breaks([0, 1, 2], closed='left')
        actual = index.get_indexer([-1, 0, 0.5, 1, 1.5, 2, 3])
        expected = [-1, 0, 0, 1, 1, -1, -1]
        self.assert_numpy_array_equal(actual, expected)

        actual = self.index.get_indexer(index[:1])
        expected = [0]
        self.assert_numpy_array_equal(actual, expected)

        self.assertRaises(KeyError, self.index.get_indexer, index)

    def test_get_indexer_subintervals(self):
        # return indexers for wholly contained subintervals
        target = IntervalIndex.from_breaks(np.linspace(0, 2, 5))
        actual = self.index.get_indexer(target)
        expected = [0, 0, 1, 1]
        self.assert_numpy_array_equal(actual, expected)

        target = IntervalIndex.from_breaks([0, 0.67, 1.33, 2])
        self.assertRaises(KeyError, self.index.get_indexer, target)

        actual = self.index.get_indexer(target[[0, -1]])
        expected = [0, 1]
        self.assert_numpy_array_equal(actual, expected)

        target = IntervalIndex.from_breaks([0, 0.33, 0.67, 1], closed='left')
        actual = self.index.get_indexer(target)
        expected = [0, 0, 0]
        self.assert_numpy_array_equal(actual, expected)

    def test_contains(self):
        self.assertNotIn(0, self.index)
        self.assertIn(0.5, self.index)
        self.assertIn(2, self.index)

        self.assertIn(Interval(0, 1), self.index)
        self.assertIn(Interval(0, 2), self.index)
        self.assertIn(Interval(0, 0.5), self.index)
        self.assertNotIn(Interval(3, 5), self.index)
        self.assertNotIn(Interval(-1, 0, closed='left'), self.index)

    def test_non_contiguous(self):
        index = IntervalIndex.from_tuples([(0, 1), (2, 3)])
        target = [0.5, 1.5, 2.5]
        actual = index.get_indexer(target)
        expected = [0, -1, 1]
        self.assert_numpy_array_equal(actual, expected)

        self.assertNotIn(1.5, index)

    def test_union(self):
        other = IntervalIndex([2], [3])
        expected = IntervalIndex(range(3), range(1, 4))
        actual = self.index.union(other)
        self.assertTrue(expected.equals(actual))

        actual = other.union(self.index)
        self.assertTrue(expected.equals(actual))

    def test_intersection(self):
        other = IntervalIndex.from_breaks([1, 2, 3])
        expected = IntervalIndex.from_breaks([1, 2])
        actual = self.index.intersection(other)
        self.assertTrue(expected.equals(actual))

    def test_isin(self):
        actual = self.index.isin(self.index)
        self.assert_numpy_array_equal([True, True], actual)

        actual = self.index.isin(self.index[:1])
        self.assert_numpy_array_equal([True, False], actual)

    def test_comparison(self):
        actual = self.index > 0
        expected = [True, True]
        self.assert_numpy_array_equal(actual, expected)

        actual = self.index <= 1
        expected = [False, False]
        self.assert_numpy_array_equal(actual, expected)

        actual = self.index < 1.5
        expected = [True, False]
        self.assert_numpy_array_equal(actual, expected)
        actual = self.index <= 1.5
        self.assert_numpy_array_equal(actual, expected)

        actual = Interval(0, 1) < self.index
        expected = [False, True]
        self.assert_numpy_array_equal(actual, expected)

        actual = Interval(0.5, 1.5) > self.index
        expected = [False, False]
        self.assert_numpy_array_equal(actual, expected)
        actual = self.index > Interval(0.5, 1.5)
        self.assert_numpy_array_equal(actual, expected)

        actual = self.index == self.index
        expected = [True, True]
        self.assert_numpy_array_equal(actual, expected)
        actual = self.index <= self.index
        self.assert_numpy_array_equal(actual, expected)
        actual = self.index >= self.index
        self.assert_numpy_array_equal(actual, expected)

        actual = self.index < self.index
        expected = [False, False]
        self.assert_numpy_array_equal(actual, expected)
        actual = self.index > self.index
        self.assert_numpy_array_equal(actual, expected)

        actual = self.index == IntervalIndex.from_breaks([0, 1, 2], 'left')
        self.assert_numpy_array_equal(actual, expected)

        self.assertRaises(ValueError, lambda: self.index > np.arange(3))
        # numpy<1.10 incorrectly raises an AttributeError instead of ValueError
        # for np.arange(2) == np.arange(3), so allow any exception here:
        self.assertRaises(Exception, lambda: self.index == np.arange(3))

    @unittest.skip('no arithmetic yet')
    def test_math(self):
        # add, subtract, multiply, divide with scalers should be OK
        actual = 2 * self.index + 1
        expected = IntervalIndex.from_breaks((2 * np.arange(3) + 1))
        self.assertTrue(expected.equals(actual))

        actual = self.index / 2.0 - 1
        expected = IntervalIndex.from_breaks((np.arange(3) / 2.0 - 1))
        self.assertTrue(expected.equals(actual))

        with self.assertRaises(TypeError):
            # doesn't make sense to add two IntervalIndex objects
            self.index + self.index

    def test_datetime(self):
        dates = pd.date_range('2000', periods=3)
        idx = IntervalIndex.from_breaks(dates)

        self.assert_numpy_array_equal(idx.left, dates[:2])
        self.assert_numpy_array_equal(idx.right, dates[-2:])

        expected = pd.date_range('2000-01-01T12:00', periods=2)
        self.assert_numpy_array_equal(idx.mid, expected)

        self.assertIn('2000-01-01T12', idx)

        target = pd.date_range('1999-12-31T12:00', periods=7, freq='12H')
        actual = idx.get_indexer(target)
        expected = [-1, -1, 0, 0, 1, 1, -1]
        self.assert_numpy_array_equal(actual, expected)

    @unittest.skip('no arithmetic yet')
    def test_datetime_math(self):

        expected = IntervalIndex(pd.date_range('2000-01-02', periods=3))
        actual = idx + pd.to_timedelta(1, unit='D')
        self.assertTrue(expected.equals(actual))

    # TODO: other set operations (left join, right join, intersection),
    # set operations with conflicting IntervalIndex objects or other dtypes,
    # groupby, cut, reset_index...
