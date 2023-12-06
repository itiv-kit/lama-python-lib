import unittest
from lama.checkpoints import lama_compare_checkpoint, lama_create_checkpoint
import pickle
import pandas
import os
import warnings


# https://stackoverflow.com/questions/3892218/how-to-test-with-pythons-unittest-that-a-warning-has-been-thrown
class WarningTestMixin(object):
    'A test which checks if the specified warning was raised'

    def assertWarns(self, warning, callable, *args, **kwds):
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always')
            result = callable(*args, **kwds)
            self.assertTrue(any(item.category == warning for item in warning_list))


class TestLAMACheckpointFunction(unittest.TestCase, WarningTestMixin):

    @classmethod
    def setUpClass(cls):
        golden_int = 5
        lama_create_checkpoint(golden_int, 'golden_int.pkl')

        golden_str = "test"
        lama_create_checkpoint(golden_str, 'golden_str.pkl')

        golden_list = list(range(0, 50, 2))
        lama_create_checkpoint(golden_list, 'golden_list.pkl')

        golden_pandas = pandas.DataFrame(
            {
                "col1": ["a", "a", "b", "b", "a"],
                "col2": [1.0, 2.0, 3.0, 6.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0]
            },
            columns=["col1", "col2", "col3"],
        )
        lama_create_checkpoint(golden_pandas, 'golden_pandas.pkl')

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove('checkpoints/golden_int.pkl')
        os.remove('checkpoints/golden_str.pkl')
        os.remove('checkpoints/golden_list.pkl')
        os.remove('checkpoints/golden_pandas.pkl')

    def test_wrong_type(self):
        with self.assertRaises(AssertionError) as ctx:
            lama_compare_checkpoint(unittest.TestCase(), 'golden_int.pkl')
        print(ctx.exception)

    def test_int(self):
        self.assertTrue(lama_compare_checkpoint(5, 'golden_int.pkl'))

    def test_wrong_int(self):
        self.assertWarns(UserWarning, lama_compare_checkpoint, 4, 'golden_int.pkl')

    def test_list(self):
        self.assertTrue(
            lama_compare_checkpoint(list(range(0, 50, 2)), 'golden_list.pkl'))

    def test_wrong_list(self):
        l = list(range(0, 50, 2))
        for i in [4, 10, 14]:
            l.remove(i)
        self.assertWarns(UserWarning, lama_compare_checkpoint, l, 'golden_list.pkl')

    def test_wrong_list2(self):
        l = list(range(0, 50, 2))
        l[10] = 100
        l[20] = 120
        self.assertWarns(UserWarning, lama_compare_checkpoint, l, 'golden_list.pkl')

    def test_dataframe(self):
        df = pandas.DataFrame(
            {
                "col1": ["a", "a", "b", "b", "a"],
                "col2": [1.0, 2.0, 3.0, 6.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0]
            },
            columns=["col1", "col2", "col3"],
        )
        self.assertTrue(lama_compare_checkpoint(df, 'golden_pandas.pkl'))

    def test_wrong_dataframe(self):
        df = pandas.DataFrame(
            {
                "col1": ["a", "a", "c", "b", "a"],
                "col2": [1.0, 2.0, 8.0, 6.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0]
            },
            columns=["col1", "col2", "col3"],
        )
        self.assertWarns(UserWarning, lama_compare_checkpoint, df, 'golden_pandas.pkl')


if __name__ == '__main__':
    unittest.main()

