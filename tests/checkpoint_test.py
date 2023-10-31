import unittest
from lama.checkpoints import lama_compare_checkpoint
import pickle
import pandas
import os


class TestLAMACheckpointFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        golden_int = 5
        with open('golden_int.pkl', 'wb') as f:
            pickle.dump(golden_int, f)

        golden_str = "test"
        with open('golden_str.pkl', 'wb') as f:
            pickle.dump(golden_str, f)

        golden_list = list(range(0,50,2))
        with open('golden_list.pkl', 'wb') as f:
            pickle.dump(golden_list, f)

        golden_pandas = pandas.DataFrame(
            {
                "col1": ["a", "a", "b", "b", "a"],
                "col2": [1.0, 2.0, 3.0, 6.0, 5.0],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0]
            },
            columns=["col1", "col2", "col3"],
        )
        with open('golden_pandas.pkl', 'wb') as f:
            pickle.dump(golden_pandas, f)

        
    @classmethod
    def tearDownClass(cls) -> None:
        os.remove('golden_int.pkl')
        os.remove('golden_str.pkl')
        os.remove('golden_list.pkl')

    def test_wrong_type(self):
        with self.assertRaises(AssertionError) as ctx:
            lama_compare_checkpoint(unittest.TestCase(), 'golden_int.pkl')
        print(ctx.exception)

    def test_int(self):
        self.assertTrue(lama_compare_checkpoint(5, 'golden_int.pkl'))

    def test_wrong_int(self):
        with self.assertRaises(AssertionError) as ctx:
            lama_compare_checkpoint(4, 'golden_int.pkl')
        print(ctx.exception)
        
    def test_list(self):
        self.assertTrue(lama_compare_checkpoint(list(range(0,50,2)), 'golden_list.pkl'))

    def test_wrong_list(self):
        l = list(range(0,50,2))
        for i in [4,10,14]:
            l.remove(i)
        with self.assertRaises(AssertionError) as ctx:
            lama_compare_checkpoint(l, 'golden_list.pkl')
        print(ctx.exception)

    def test_wrong_list2(self):
        l = list(range(0,50,2))
        l[10] = 100
        l[20] = 120
        with self.assertRaises(AssertionError) as ctx:
            lama_compare_checkpoint(l, 'golden_list.pkl')
        print(ctx.exception)
    
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
        with self.assertRaises(UserWarning) as ctx:
            lama_compare_checkpoint(df, 'golden_pandas.pkl')
        print(ctx.exception)

if __name__ == '__main__':
    unittest.main()