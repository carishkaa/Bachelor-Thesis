#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
from functions.classificator import classify
from functions.data_preprocess import preprocess_for_classification, compensation


def experiment_iteration(STA_range, n_repeats, gen_origin, gen_expected, gen_IS, n_genSTA, distribution):
    n_STA = STA_range[-1]  # maximal number of STA patients

    # arrays for AUC results
    mean_auc_results_origin = np.zeros((n_repeats, len(STA_range)))
    mean_auc_results_expected = np.zeros((n_repeats, len(STA_range)))
    mean_auc_results_filtered = np.zeros((n_repeats, len(STA_range)))

    for j in range(n_repeats):
        STA_samples_random_index = random.sample(range(n_genSTA), n_genSTA - n_STA)
        a = gen_origin.drop(gen_origin.columns[STA_samples_random_index], axis=1)
        b = gen_expected.drop(gen_expected.columns[STA_samples_random_index], axis=1)
        c = gen_IS.drop(gen_IS.index[STA_samples_random_index], axis=0)

        patient_labels = a.columns

        for i in range(len(STA_range)):
            # leave only STA_range[i] STA patients
            a_tmp = a.drop(patient_labels[range(n_STA - STA_range[i])], axis=1)
            b_tmp = b.drop(patient_labels[range(n_STA - STA_range[i])], axis=1)
            c_tmp = c.drop(patient_labels[range(n_STA - STA_range[i])], axis=0)
            try:
                d, e, f = experiment(a_tmp, b_tmp, c_tmp, n_CR=15, distribution=distribution)
                mean_auc_results_origin[j, i] = d.mean()
                mean_auc_results_expected[j, i] = e.mean()
                mean_auc_results_filtered[j, i] = f.mean()
            except: # when STA number is too small, there are the PerfectSeparation error
                print('Exception when ', STA_range[i], ' STA samples.')
                mean_auc_results_origin[j, i] = np.nan
                mean_auc_results_expected[j, i] = np.nan
                mean_auc_results_filtered[j, i] = np.nan
    return mean_auc_results_origin, mean_auc_results_expected, mean_auc_results_filtered



def experiment(generated_counts, generated_expected_counts, generated_IS, n_CR: int, distribution):
    # counts generated with IS influence
    counts, patients_labels, _ = preprocess_for_classification(generated_counts)

    # counts generated with similar beta0, but without IS influence
    expected_counts, _, _ = preprocess_for_classification(generated_expected_counts)

    # IS compensation
    t = generated_counts.shape[0]
    filtered_counts, OT_CR_IS, learned_beta = compensation(generated_counts, generated_IS, t, n_CR, distribution)

    columns = ["transcript" + str(num) for num in range(counts.shape[1])]
    y = patients_labels

    # COUNTS
    X = pd.DataFrame(data=counts, columns=columns)
    y = pd.DataFrame(data=y)
    y.head()
    results1 = classify(X, y, n_splits=12, n_repeats=1)

    # EXPECTED
    X = pd.DataFrame(data=expected_counts, columns=columns)
    y = pd.DataFrame(data=y)
    y.head()
    results2 = classify(X, y, n_splits=12, n_repeats=1)

    # FILTERED
    X = pd.DataFrame(data=filtered_counts, columns=columns)
    y = pd.DataFrame(data=y)
    y.head()
    results3 = classify(X, y, n_splits=12, n_repeats=1)

    return results1, results2, results3
