#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compensation(counts, IS, t, n_CR, distribution):
    """
    IS compensation.

    :param counts: Counts to be compensated.
    :param IS: Array of immunosuppression intake (0 or 1).
    :param t: Number of transcripts.
    :param distribution: "Poisson" or "Negative binomial".
    :param n_CR: Number of samples in CR class. First n_CR elements of counts must be of CR class.
    """
    # leave only STA
    STA_patients = counts.columns.str.startswith('STA')
    STA_counts = counts.loc[:, STA_patients].to_numpy()
    STA_IS = IS.loc[STA_patients]

    # try to learn the beta1 parameters from STA class
    # TODO: save learned_beta and use it to speed up code
    learned_beta = np.zeros((3, t))
    for ind in range(t):
        df = pd.DataFrame({"x1": STA_IS['IS1'].values.tolist(),
                           "x2": STA_IS['IS2'].values.tolist(),
                           "x3": STA_IS['IS3'].values.tolist(),
                           "y": STA_counts[ind]})
        if distribution == "Poisson":
            mp = sm.formula.glm("y ~ x1 + x2 + x3", family=sm.families.Poisson(), data=df).fit()
        elif distribution == "Negative binomial":
            mp = sm.formula.glm("y ~ x1 + x2 + x3", family=sm.families.NegativeBinomial(), data=df).fit()
        else:
            print("Unknown distribution")
            return
        learned_beta[:, ind] = np.array([mp.params.x1, mp.params.x2, mp.params.x3])

    # leave only CR and OT classes for classification
    OT_CR_counts = counts.loc[:, ~STA_patients].to_numpy(dtype=float)
    OT_CR_IS = IS.loc[~STA_patients].values

    # filter out IS from CR
    OT_CR_counts[:, :n_CR] = OT_CR_counts[:, :n_CR] / np.exp(np.dot(OT_CR_IS[:n_CR], learned_beta).transpose())
    OT_CR_counts = OT_CR_counts.transpose()
    return OT_CR_counts, OT_CR_IS, learned_beta


def preprocess_for_classification(samples):
    # leave only OT_ and CR_ patients
    STA_counts = samples.columns.str.startswith('STA')
    samples = samples.loc[:, ~STA_counts]

    # recode patient status to numeric (OT = 0, CR = 1)
    patients = samples.columns
    patients_labels = pd.DataFrame({'status': [p[:2] for p in patients]})

    patients_labels = patients_labels.replace({'OT': 0, 'CR': 1})
    labels = ['OT', 'CR']

    # transpose
    samples = samples.transpose()
    samples = samples.reset_index(drop=True)
    return samples, patients_labels, labels
