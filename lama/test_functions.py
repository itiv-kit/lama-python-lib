import unittest
from unittest import TestCase

import numpy as np
from numbers import Integral

RANDOM_STATE = 42

OK_MSG = 'Your data looks alright, you can continue with the workbook.' 
ERR_MSG = 'Oops, there is still an issue. Please fix your implementation before continuing with the workbook.'


# Task 2 - Supervised Learning
class SupervisedLearning_Tests(TestCase):

    def test_train_age_bins(self, train):
        expected_val = {0: 78, 1: 433, 2: 210, 3: 106, 4: 42, 5: 22}
        self.assertTrue(train["Age_bin"].dtype == 'int64', msg=ERR_MSG)
        self.assertEqual(len(set(train["Age_bin"])), 6, msg=ERR_MSG)
        self.assertDictEqual(train['Age_bin'].value_counts().to_dict(), expected_val, msg=ERR_MSG)
        print(OK_MSG)

    def test_train_feature_selection(self, train):
        self.assertEqual(train.columns.size, 14, msg=ERR_MSG) # TODO: update size
        self.assertNotIn('Name', train.columns.to_list(), msg=ERR_MSG)
        self.assertNotIn('Ticket', train.columns.to_list(), msg=ERR_MSG)
        print(OK_MSG)

    def test_decision_tree_acc(self, accuracy_score):
        self.assertAlmostEqual(accuracy_score, 0.7821, places=4, msg=ERR_MSG)
        print(OK_MSG)

    def test_cross_validation_acc(self, acc1):
        self.assertAlmostEqual(acc1, 0.7733, places=4, msg=ERR_MSG)
        print(OK_MSG)

    def test_best_param(self, best_param):
        self.assertIsInstance(best_param, dict, msg=ERR_MSG)
        key = list(best_param.keys())[0]
        self.assertEqual(best_param.get(key), 3, msg=ERR_MSG)
        print(OK_MSG)

    def test_GridSearchCV_acc(self, acc2):
        self.assertAlmostEqual(acc2, 0.80696, places=4, msg=ERR_MSG)
        print(OK_MSG)

    def test_best_neighbors(self, best_neighbors):
        self.assertEqual(best_neighbors, 6, msg=ERR_MSG)
        print(OK_MSG)
    
# Task 3 - Unsupervised Learning
class UnsupervisedLearning_Tests(TestCase):

    def test_labels_blob(self, labels_blob):
        n_clusters = 4
        self.assertTrue(labels_blob.size > 0, msg=ERR_MSG)
        self.assertIsInstance(labels_blob, np.ndarray, msg=ERR_MSG)
        self.assertTrue(all(val in set(labels_blob) for val in range(n_clusters)), msg=ERR_MSG)
        print(OK_MSG)

    def test_labels_kmeans(self, labels_kmeans):
        self.assertIsInstance(labels_kmeans, np.ndarray, msg=ERR_MSG)
        self.assertTrue(labels_kmeans.size > 0, msg=ERR_MSG)
        self.assertTrue(len(labels_kmeans.shape) == 1, msg=ERR_MSG)
        print(OK_MSG)
    
    def test_calculate_k_loss(self, calculate_k_loss):
        expected_losses = [64.7523, 45.6046, 35.1672, 25.4615]
        np.random.seed(RANDOM_STATE)
        X = np.random.randn(15, 5)
        _, losses = calculate_k_loss(X, k_min=1, k_max=5)
        losses_rounded = [round(loss, 4) for loss in losses]
        self.assertListEqual(losses_rounded, expected_losses, msg=ERR_MSG)
        print(OK_MSG)

    def test_best_knn_score(self, best_knn_score):
        self.assertEqual(round(best_knn_score, 4), 0.8928, msg=ERR_MSG)
        print(OK_MSG)
    
    def test_best_tree_score(self, best_tree_score):
        expected_score = 0.8328 # 0.8338
        self.assertEqual(round(best_tree_score, 4), expected_score, msg=ERR_MSG)
        print(OK_MSG)

# Task 4 - Neural Networks
class NN_Tests(TestCase):
    pass

# Task 5 - Reinforcement Learning + Evolutionary Algorithms
class RL_EA_Tests(TestCase):
    pass

# Task 6 - Convolutional Neural Networks
class CNN_Tests(TestCase):
    
    def test_classes(self, classes):
        self.assertEqual(len(classes), 10, msg=ERR_MSG)
        self.assertIsInstance(classes, list)
        self.assertTrue('cat' in classes, msg=ERR_MSG) # sample check
        print(OK_MSG)



# Task 7 - Transformers and Generative AI
class Transformer_GenAI_Tests(TestCase):
    pass

if __name__ == "__main__":
    #unittest.main()
    unittest.main(argv=[''], verbosity=0, exit=False)
