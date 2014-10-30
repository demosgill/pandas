# coding=utf-8
# pylint: disable-msg=E1101,W0612

import nose

import numpy as np
import pandas as pd

from pandas import (Index, Series, DataFrame, isnull, notnull, bdate_range,
                    date_range, period_range, timedelta_range, MultiIndex)

from pandas.compat import StringIO, lrange, range, zip, u, OrderedDict, long
from pandas.core import common as com
from pandas import compat, set_option, lib
from pandas.util.testing import (assert_series_equal,
                                 assert_almost_equal,
                                 assert_frame_equal,
                                 assert_array_equivalent,
                                 ensure_clean)
import pandas.util.testing as tm

set_option('support.dynd',True)
if not com._DYND:
    raise nose.SkipTest("dynd not installed")

import dynd
from dynd import ndt, nd
import datashape
from datashape import dshape

class TestDynd(tm.TestCase):
    """ test certain attributes/method on nd arrays directly """

    _multiprocess_can_split_ = False


    def test_indexing(self):

        t = nd.array([1,2,3])
        assert_array_equivalent(t, t)

        result = t[[1,2]]
        assert_array_equivalent(result, nd.array([2,3]))

        result = t[[False,True,True]]
        assert_array_equivalent(result, nd.array([2,3]))

    def test_types(self):

        t = ndt.type('?int32')

        ds = dshape(t.dshape)
        ty = ds.measure.ty

        # to numpy dtype
        self.assertTrue(com.to_numpy_dtype(t) == np.dtype('int32'))

        # type testing
        self.assertTrue(datashape.isnumeric(ty))

        # type testing
        from datashape import integral, floating
        self.assertTrue(ty in integral.types)
        self.assertFalse(ty in floating.types)

        t = ndt.type('?void')

        # to numpy dtype
        # not implemented in datashape ATM
        self.assertTrue(com.to_numpy_dtype(t) == np.dtype('object'))

    def test_cast_dtype_to_na(self):

        # safe
        t = ndt.type('3 * int64')
        self.assertEqual(com.cast_dtype_to_na(t,kind='safe'),ndt.type('3 * ?int64'))
        t = ndt.type('3 * int32')
        self.assertEqual(com.cast_dtype_to_na(t,kind='safe'),ndt.type('3 * ?int32'))
        t = ndt.type('3 * ?int32')
        self.assertEqual(com.cast_dtype_to_na(t,kind='safe'),ndt.type('3 * ?int32'))

        # opportunistic
        t = ndt.type('3 * float32')
        self.assertEqual(com.cast_dtype_to_na(t,kind='safe'),ndt.type('3 * ?int32'))

        # same
        t = ndt.type('3 * int64')
        self.assertEqual(com.cast_dtype_to_na(t,kind='same'),ndt.type('3 * ?int64'))
        t = ndt.type('3 * ?int64')
        self.assertEqual(com.cast_dtype_to_na(t,kind='same'),ndt.type('3 * ?int64'))
        t = ndt.type('3 * float64')
        self.assertEqual(com.cast_dtype_to_na(t,kind='same'),ndt.type('3 * ?float64'))
        t = ndt.type('3 * ?float64')
        self.assertEqual(com.cast_dtype_to_na(t,kind='same'),ndt.type('3 * ?float64'))

    def test_cast_dtype_from_na(self):

        t = ndt.type('3 * int64')
        self.assertEqual(com.cast_dtype_from_na(t),ndt.type('3 * int64'))
        t = ndt.type('3 * int32')
        self.assertEqual(com.cast_dtype_from_na(t),ndt.type('3 * int32'))
        t = ndt.type('3 * ?int32')
        self.assertEqual(com.cast_dtype_from_na(t),ndt.type('3 * int32'))

    def test_cast_to_dynd(self):

        assert_array_equivalent(com.cast_to_dynd(np.array([1,2,3],dtype='int64')),
                                nd.array([1,2,3],type='3 * ?int64'))
        assert_array_equivalent(com.cast_to_dynd(np.array([1.,2.,3.],dtype='float64')),
                                nd.array([1,2,3],type='3 * ?int64'))
        assert_array_equivalent(com.cast_to_dynd(np.array([1.,np.nan,3.],dtype='float64')),
                                nd.array([1,None,3],type='3 * ?int64'))

        # no conversion
        assert_array_equivalent(com.cast_to_dynd(np.array([1.,np.nan,3.5],dtype='float64')),
                                nd.array([1,np.nan,3.5],dtype='float64'))

        assert_array_equivalent(com.cast_to_dynd([None,None],dtype='int64'),
                                nd.array([None,None],dtype='?int64'))

    def test_cast_to_numpy(self):

        # test conversions to numpy
        assert_array_equivalent(com.cast_to_numpy(nd.array([1,2,3],dtype='int64')),
                                np.array([1,2,3],dtype='int64'))

        assert_array_equivalent(com.cast_to_numpy(nd.array([1,2,3],dtype='?int64')),
                                np.array([1,2,3],dtype='int64'))


class TestBasic(tm.TestCase):
    _multiprocess_can_split_ = False

    def test_config_option(self):

        set_option('support.dynd',False)
        set_option('support.dynd',True)

    def test_api_compat(self):

        arr = nd.array([1,2,3],type='3 * ?int32')
        self.assertTrue(arr.type.dtype == ndt.type('?int32'))
        self.assertTrue(arr.dtype == ndt.type('?int32'))
        self.assertTrue(arr.ndim == 1)

    def test_formatting(self):

        s = Series([1,2,3])
        result = str(s)
        expected = "0   1\n1   2\n2   3\ndtype: int64"
        self.assertEqual(result, expected)

        s = Series([1,None,3])
        result = str(s)
        expected = "0     1\n1   NaN\n2     3\ndtype: int64"
        self.assertEqual(result, expected)

    def test_na(self):

        s = Series([1,2,3])
        result = s.isnull()
        tm.assert_series_equal(result, Series(False,index=s.index))

        s = Series([1,None,3])
        result = s.isnull()
        tm.assert_series_equal(result, Series([False,True,False],index=s.index))

        s = Series([1,None,3.])
        result = s.isnull()
        tm.assert_series_equal(result, Series([False,True,False],index=s.index))

        s = Series([1,None,3.5])
        result = s.isnull()
        tm.assert_series_equal(result, Series([False,True,False],index=s.index))

        #### FIXME: this is technically a raise_cast_failure
        # but can handle now
        s = Series([1,np.nan,3],dtype='int64')
        result = s.isnull()
        tm.assert_series_equal(result, Series([False,True,False],index=s.index))

    def test_construction(self):

        s = Series([1,2,3])
        assert_almost_equal(s.dtype, ndt.type('?int64'))

        s = Series([1,None,3])
        assert_almost_equal(s.dtype, ndt.type('?int64'))

        s = Series([1,2,3],dtype='int32')
        assert_almost_equal(s.dtype, ndt.type('?int32'))

        s = Series([],dtype='int32')
        assert_almost_equal(s.dtype, ndt.type('?int32'))

        s = Series([1,None,3.])
        assert_almost_equal(s.dtype, ndt.type('?int64'))

        s = Series([1,None,3.5])
        assert_almost_equal(s.dtype, np.dtype('float64'))

        # we should not be converting bools
        s = Series([True,True,True])
        assert_almost_equal(s.dtype, np.dtype('bool'))

        s = Series([np.nan, np.nan],dtype='int64',index=[1,2])
        assert_almost_equal(s.dtype, ndt.type('?int64'))

        s = Series([None, None],dtype='int64',index=[1,2])
        assert_almost_equal(s.dtype, ndt.type('?int64'))

    def test_indexing_get(self):

        # scalar
        s = Series([1,2,3])
        result = s.iloc[1]
        self.assertEqual(result,2)

        result = s[1]
        self.assertEqual(result,2)

        s = Series([1,np.nan,3])
        result = s.iloc[1]
        self.assertTrue(result is np.nan)

        result = s[1]
        self.assertTrue(result is np.nan)

        # slice
        s = Series([1,np.nan,3])
        expected = Series([np.nan,3],index=[1,2])
        result = s[1:3]
        assert_series_equal(result, expected)

        # boolean
        result = s[[False,True,True]]
        assert_series_equal(result, expected)

    def test_indexing_set(self):

        # scalar with coercion
        s = Series([1,2,3])
        s.iloc[1] = 4
        result = s.iloc[1]

        self.assertTrue(result is 4)
        s = Series([1,2,3])
        s.iloc[1] = np.nan
        result = s.iloc[1]
        self.assertTrue(result is np.nan)

        expected = Series([4,4],index=[1,2])
        s = Series([1,2,3])
        s.iloc[1:3] = 4
        result = s.iloc[1:3]
        assert_series_equal(result, expected)

        expected = Series([1, np.nan, np.nan])
        s = Series([1,2,3])
        s.iloc[1:3] = np.nan
        assert_series_equal(s, expected)

        s = Series([1,2,3])
        s.iloc[[False,True,True]] = np.nan
        assert_series_equal(s, expected)

        # array with coercion
        expected = Series([1, np.nan, 1])
        s = Series([1,2,3])
        s.iloc[1:3] = [np.nan,1]
        assert_series_equal(s, expected)

        s = Series([1,2,3])
        s.iloc[[False,True,True]] = [np.nan, 1]
        assert_series_equal(s, expected)

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
