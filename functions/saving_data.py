#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def get_empty_auc_dataset():
    """
    data[number_of_transcripts][IS_number_of_effected_transcripts][IS_effect][n_class][c]

    :return:
    """
    data = dict()

    transcripts = [10, 100, 1000, 10000]
    class_effect_n_transcripts = [0, 2, 5, 10, 20]
    class_effect = [0.2, 0.3, 0.4]  # mean of normal distribution

    IS_effect_n_transcripts = [2, 5, 10, 20]
    IS_effect = [0.3, 0.5, 0.9]  # coefficients beta1..beta3

    for t in transcripts:
        data[t] = dict()
        for n_IS in IS_effect_n_transcripts:
            data[t][n_IS] = dict()
            for i in IS_effect:
                data[t][n_IS][i] = dict()
                for n_class in class_effect_n_transcripts:
                    data[t][n_IS][i][n_class] = dict()
                    for c in class_effect:
                        data[t][n_IS][i][n_class][c] = {'STA_range': [],
                                                        'auc': np.array([])}
    return data
