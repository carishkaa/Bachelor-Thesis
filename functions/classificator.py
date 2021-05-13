#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


def classify(X, y, n_splits, n_repeats):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats).split(X, y)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SGDClassifier(alpha=0.02, loss='log', penalty='elasticnet', l1_ratio=1))
    ])
    score_value = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    mean_AUC_by_fold = np.mean(score_value.reshape(-1, n_splits), axis=1)
    return mean_AUC_by_fold


