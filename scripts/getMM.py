import numpy as np
models=["Cat Boost","Perceptron","K-Neighbours Classifier","Random Forests Classifier","SVC","XGBoost"]
t=["""TEST NUMBER 1 Random Seed = 42
Hiding  24  habitable(3233,163,117,2031,2014,1845,1137,703,3716,153,3922,2883,130,128,1205,2156,2223,151,2542,1604,3133,3115,3741,2880, )
BEST SCORE: 0.9857954545454546
PARAMS: {'logging_level': 'Silent', 'loss_function': 'MultiClass', 'depth': 3, 'learning_rate': 0.1, 'iterations': 50}
**********

TEST NUMBER 2 Random Seed = 52
Hiding  24  habitable(2014,130,3962,2882,2547,2189,2503,128,1137,1227,2883,1604,2541,3233,114,152,3132,3742,117,1424,163,1205,1845,2542, )
BEST SCORE: 0.96875
PARAMS: {'logging_level': 'Silent', 'loss_function': 'Logloss', 'depth': 2, 'learning_rate': 0.01, 'iterations': 200}
**********

TEST NUMBER 3 Random Seed = 62
Hiding  24  habitable(2902,1147,151,2014,2547,2135,3962,2223,1205,3233,1424,1227,114,2156,1845,986,128,3744,2441,2541,117,3741,2129,3115, )
BEST SCORE: 0.9403409090909092
PARAMS: {'logging_level': 'Silent', 'loss_function': 'Logloss', 'depth': 2, 'learning_rate': 0.01, 'iterations': 200}
**********

TEST NUMBER 4 Random Seed = 72
Hiding  24  habitable(151,3133,1205,2156,3962,3115,3742,2223,2129,3744,986,2316,2014,2547,1147,117,2097,3741,3132,114,2155,152,3716,1845, )
BEST SCORE: 0.9545454545454546
PARAMS: {'logging_level': 'Silent', 'loss_function': 'MultiClassOneVsAll', 'depth': 3, 'learning_rate': 0.1, 'iterations': 50}
**********

TEST NUMBER 5 Random Seed = 82
Hiding  24  habitable(986,2547,2882,2031,1205,1147,2097,1227,128,2902,3233,3962,163,2021,2441,1424,1137,3744,130,2883,2135,3742,2156,3743, )
BEST SCORE: 0.875
PARAMS: {'logging_level': 'Silent', 'loss_function': 'Logloss', 'depth': 3, 'learning_rate': 0.1, 'iterations': 10}""",
"""

**********

TEST NUMBER 0 Random Seed = 42
Hiding  24  habitable(3233,163,117,2031,2014,1845,1137,703,3716,153,3922,2883,130,128,1205,2156,2223,151,2542,1604,3133,3115,3741,2880, )
BEST SCORE: 0.8607954545454546
PARAMS: {'alpha': 0.001, 'class_weight': None, 'early_stopping': True, 'eta0': 50, 'fit_intercept': True, 'max_iter': 500, 'n_iter_no_change': 13, 'n_jobs': -1, 'penalty': 'l2', 'random_state': 0, 'shuffle': True, 'tol': 1e-05, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
**********

TEST NUMBER 1 Random Seed = 52
Hiding  24  habitable(2014,130,3962,2882,2547,2189,2503,128,1137,1227,2883,1604,2541,3233,114,152,3132,3742,117,1424,163,1205,1845,2542, )
BEST SCORE: 0.84375
PARAMS: {'alpha': 0.01, 'class_weight': None, 'early_stopping': True, 'eta0': 1, 'fit_intercept': True, 'max_iter': 500, 'n_iter_no_change': 13, 'n_jobs': -1, 'penalty': 'l2', 'random_state': 0, 'shuffle': True, 'tol': 1e-05, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
**********

TEST NUMBER 2 Random Seed = 62
Hiding  24  habitable(2902,1147,151,2014,2547,2135,3962,2223,1205,3233,1424,1227,114,2156,1845,986,128,3744,2441,2541,117,3741,2129,3115, )
BEST SCORE: 0.8210227272727273
PARAMS: {'alpha': 0.01, 'class_weight': None, 'early_stopping': True, 'eta0': 1, 'fit_intercept': True, 'max_iter': 500, 'n_iter_no_change': 20, 'n_jobs': -1, 'penalty': 'l2', 'random_state': 0, 'shuffle': True, 'tol': 1e-05, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
**********

TEST NUMBER 3 Random Seed = 72
Hiding  24  habitable(151,3133,1205,2156,3962,3115,3742,2223,2129,3744,986,2316,2014,2547,1147,117,2097,3741,3132,114,2155,152,3716,1845, )
BEST SCORE: 0.8494318181818182
PARAMS: {'alpha': 0.001, 'class_weight': None, 'early_stopping': True, 'eta0': 1, 'fit_intercept': True, 'max_iter': 500, 'n_iter_no_change': 20, 'n_jobs': -1, 'penalty': 'l2', 'random_state': 0, 'shuffle': True, 'tol': 1e-05, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
**********

TEST NUMBER 4 Random Seed = 82
Hiding  24  habitable(986,2547,2882,2031,1205,1147,2097,1227,128,2902,3233,3962,163,2021,2441,1424,1137,3744,130,2883,2135,3742,2156,3743, )
BEST SCORE: 0.8522727272727273
PARAMS: {'alpha': 0.001, 'class_weight': None, 'early_stopping': True, 'eta0': 1, 'fit_intercept': True, 'max_iter': 500, 'n_iter_no_change': 20, 'n_jobs': -1, 'penalty': 'l2', 'random_state': 0, 'shuffle': True, 'tol': 1e-05, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}

""",
"""**********

TEST NUMBER 1 Random Seed = 42
Hiding  24  habitable(3233,163,117,2031,2014,1845,1137,703,3716,153,3922,2883,130,128,1205,2156,2223,151,2542,1604,3133,3115,3741,2880, )
BEST SCORE: 0.7869318181818181
PARAMS: {'algorithm': 'auto', 'leaf_size': 10, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 7, 'p': 5, 'weights': 'uniform'}
**********

TEST NUMBER 2 Random Seed = 52
Hiding  24  habitable(2014,130,3962,2882,2547,2189,2503,128,1137,1227,2883,1604,2541,3233,114,152,3132,3742,117,1424,163,1205,1845,2542, )
BEST SCORE: 0.7471590909090909
PARAMS: {'algorithm': 'auto', 'leaf_size': 10, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 7, 'p': 2, 'weights': 'uniform'}
**********

TEST NUMBER 3 Random Seed = 62
Hiding  24  habitable(2902,1147,151,2014,2547,2135,3962,2223,1205,3233,1424,1227,114,2156,1845,986,128,3744,2441,2541,117,3741,2129,3115, )
BEST SCORE: 0.7244318181818181
PARAMS: {'algorithm': 'auto', 'leaf_size': 10, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 7, 'p': 5, 'weights': 'uniform'}
**********

TEST NUMBER 4 Random Seed = 72
Hiding  24  habitable(151,3133,1205,2156,3962,3115,3742,2223,2129,3744,986,2316,2014,2547,1147,117,2097,3741,3132,114,2155,152,3716,1845, )
BEST SCORE: 0.7471590909090909
PARAMS: {'algorithm': 'auto', 'leaf_size': 10, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 5, 'p': 5, 'weights': 'uniform'}
**********

TEST NUMBER 5 Random Seed = 82
Hiding  24  habitable(986,2547,2882,2031,1205,1147,2097,1227,128,2902,3233,3962,163,2021,2441,1424,1137,3744,130,2883,2135,3742,2156,3743, )
BEST SCORE: 0.7386363636363636
PARAMS: {'algorithm': 'auto', 'leaf_size': 10, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 7, 'p': 5, 'weights': 'uniform'}""",
"""**********

TEST NUMBER 1 Random Seed = 42
Hiding  24  habitable(3233,163,117,2031,2014,1845,1137,703,3716,153,3922,2883,130,128,1205,2156,2223,151,2542,1604,3133,3115,3741,2880, )
BEST SCORE: 0.9772727272727273
PARAMS: {'bootstrap': True, 'class_weight': None, 'criterion': 'entropy', 'max_depth': 6, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 15, 'n_jobs': -1, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
**********

TEST NUMBER 2 Random Seed = 52
Hiding  24  habitable(2014,130,3962,2882,2547,2189,2503,128,1137,1227,2883,1604,2541,3233,114,152,3132,3742,117,1424,163,1205,1845,2542, )
BEST SCORE: 0.9005681818181818
PARAMS: {'bootstrap': True, 'class_weight': None, 'criterion': 'entropy', 'max_depth': 6, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 43, 'n_jobs': -1, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
**********

TEST NUMBER 3 Random Seed = 62
Hiding  24  habitable(2902,1147,151,2014,2547,2135,3962,2223,1205,3233,1424,1227,114,2156,1845,986,128,3744,2441,2541,117,3741,2129,3115, )
BEST SCORE: 0.8806818181818181
PARAMS: {'bootstrap': True, 'class_weight': None, 'criterion': 'gini', 'max_depth': 6, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 8, 'n_jobs': -1, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
**********

TEST NUMBER 4 Random Seed = 72
Hiding  24  habitable(151,3133,1205,2156,3962,3115,3742,2223,2129,3744,986,2316,2014,2547,1147,117,2097,3741,3132,114,2155,152,3716,1845, )
BEST SCORE: 0.9005681818181819
PARAMS: {'bootstrap': True, 'class_weight': None, 'criterion': 'entropy', 'max_depth': 6, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 66, 'n_jobs': -1, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
**********

TEST NUMBER 5 Random Seed = 82
Hiding  24  habitable(986,2547,2882,2031,1205,1147,2097,1227,128,2902,3233,3962,163,2021,2441,1424,1137,3744,130,2883,2135,3742,2156,3743, )
BEST SCORE: 0.7897727272727273
PARAMS: {'bootstrap': True, 'class_weight': None, 'criterion': 'gini', 'max_depth': 6, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 6, 'n_jobs': -1, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}""",
"""**********

TEST NUMBER 1 Random Seed = 42
Hiding  24  habitable(3233,163,117,2031,2014,1845,1137,703,3716,153,3922,2883,130,128,1205,2156,2223,151,2542,1604,3133,3115,3741,2880, )
BEST SCORE: 0.8238636363636364
PARAMS: {'C': 0.001, 'cache_size': 200, 'class_weight': None, 'coef0': 0, 'decision_function_shape': 'ovo', 'degree': 1, 'gamma': 'auto', 'kernel': 'linear', 'max_iter': -1, 'probability': False, 'random_state': None, 'shrinking': False, 'tol': 0.1, 'verbose': False}
**********

TEST NUMBER 2 Random Seed = 52
Hiding  24  habitable(2014,130,3962,2882,2547,2189,2503,128,1137,1227,2883,1604,2541,3233,114,152,3132,3742,117,1424,163,1205,1845,2542, )
BEST SCORE: 0.8238636363636364
PARAMS: {'C': 0.001, 'cache_size': 200, 'class_weight': None, 'coef0': 0, 'decision_function_shape': 'ovo', 'degree': 1, 'gamma': 'auto', 'kernel': 'linear', 'max_iter': -1, 'probability': False, 'random_state': None, 'shrinking': False, 'tol': 0.1, 'verbose': False}
**********

TEST NUMBER 3 Random Seed = 62
Hiding  24  habitable(2902,1147,151,2014,2547,2135,3962,2223,1205,3233,1424,1227,114,2156,1845,986,128,3744,2441,2541,117,3741,2129,3115, )
BEST SCORE: 0.84375
PARAMS: {'C': 0.001, 'cache_size': 200, 'class_weight': None, 'coef0': 0, 'decision_function_shape': 'ovo', 'degree': 1, 'gamma': 'auto', 'kernel': 'linear', 'max_iter': -1, 'probability': False, 'random_state': None, 'shrinking': True, 'tol': 1, 'verbose': False}
**********

TEST NUMBER 4 Random Seed = 72
Hiding  24  habitable(151,3133,1205,2156,3962,3115,3742,2223,2129,3744,986,2316,2014,2547,1147,117,2097,3741,3132,114,2155,152,3716,1845, )
BEST SCORE: 0.84375
PARAMS: {'C': 0.001, 'cache_size': 200, 'class_weight': None, 'coef0': 0, 'decision_function_shape': 'ovo', 'degree': 1, 'gamma': 'auto', 'kernel': 'linear', 'max_iter': -1, 'probability': False, 'random_state': None, 'shrinking': True, 'tol': 1, 'verbose': False}
**********

TEST NUMBER 5 Random Seed = 82
Hiding  24  habitable(986,2547,2882,2031,1205,1147,2097,1227,128,2902,3233,3962,163,2021,2441,1424,1137,3744,130,2883,2135,3742,2156,3743, )
BEST SCORE: 0.84375
PARAMS: {'C': 0.001, 'cache_size': 200, 'class_weight': None, 'coef0': 0, 'decision_function_shape': 'ovo', 'degree': 1, 'gamma': 'auto', 'kernel': 'linear', 'max_iter': -1, 'probability': False, 'random_state': None, 'shrinking': True, 'tol': 1, 'verbose': False}""",
"""

**********

TEST NUMBER 1 Random Seed = 42
Hiding  24  habitable(3233,163,117,2031,2014,1845,1137,703,3716,153,3922,2883,130,128,1205,2156,2223,151,2542,1604,3133,3115,3741,2880, )
BEST SCORE: 0.9772727272727273
PARAMS: {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.1, 'max_delta_step': 0, 'max_depth': 3, 'min_child_weight': 1, 'missing': None, 'n_estimators': 100, 'n_jobs': 1, 'nthread': None, 'objective': 'binary:logistic', 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'seed': None, 'silent': None, 'subsample': 1, 'verbosity': 0}
**********

TEST NUMBER 2 Random Seed = 52
Hiding  24  habitable(2014,130,3962,2882,2547,2189,2503,128,1137,1227,2883,1604,2541,3233,114,152,3132,3742,117,1424,163,1205,1845,2542, )
BEST SCORE: 0.9346590909090909
PARAMS: {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.1, 'max_delta_step': 0, 'max_depth': 3, 'min_child_weight': 1, 'missing': None, 'n_estimators': 100, 'n_jobs': 1, 'nthread': None, 'objective': 'binary:logistic', 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'seed': None, 'silent': None, 'subsample': 1, 'verbosity': 0}
**********

TEST NUMBER 3 Random Seed = 62
Hiding  24  habitable(2902,1147,151,2014,2547,2135,3962,2223,1205,3233,1424,1227,114,2156,1845,986,128,3744,2441,2541,117,3741,2129,3115, )
BEST SCORE: 0.7897727272727273
PARAMS: {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.1, 'max_delta_step': 0, 'max_depth': 3, 'min_child_weight': 1, 'missing': None, 'n_estimators': 100, 'n_jobs': 1, 'nthread': None, 'objective': 'binary:logistic', 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'seed': None, 'silent': None, 'subsample': 1, 'verbosity': 0}
**********

TEST NUMBER 4 Random Seed = 72
Hiding  24  habitable(151,3133,1205,2156,3962,3115,3742,2223,2129,3744,986,2316,2014,2547,1147,117,2097,3741,3132,114,2155,152,3716,1845, )
BEST SCORE: 0.9630681818181819
PARAMS: {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.1, 'max_delta_step': 0, 'max_depth': 3, 'min_child_weight': 1, 'missing': None, 'n_estimators': 100, 'n_jobs': 1, 'nthread': None, 'objective': 'binary:logistic', 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'seed': None, 'silent': None, 'subsample': 1, 'verbosity': 0}
**********

TEST NUMBER 5 Random Seed = 82
Hiding  24  habitable(986,2547,2882,2031,1205,1147,2097,1227,128,2902,3233,3962,163,2021,2441,1424,1137,3744,130,2883,2135,3742,2156,3743, )
BEST SCORE: 0.8039772727272727
PARAMS: {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.1, 'max_delta_step': 0, 'max_depth': 3, 'min_child_weight': 1, 'missing': None, 'n_estimators': 100, 'n_jobs': 1, 'nthread': None, 'objective': 'binary:logistic', 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'seed': None, 'silent': None, 'subsample': 1, 'verbosity': 0}

"""]
for n in range(len(models)):
	t[n]=t[n].split("\n")
	t[n]=[float(x.split(":")[-1].strip()) for x in t[n] if "BEST" in x]
	print(str(models[n])+":\nMean: "+str(np.mean(t[n]))+"\nMedian: "+str(np.median(t[n]))+"\nStandard Deviation: "+str(np.std(t[n])))
