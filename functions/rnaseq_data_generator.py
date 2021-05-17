#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.stats as stats
import random


def rna_seq_generator(
        n_transcripts: int,
        distribution: str,
        n_STA: int, n_CR: int, n_OT: int,
        n_IS_effect: int, IS_effect: tuple,
        is_class_effect: bool, n_class_effect: int, class_effect: tuple,
        result_file_name: str = ""
):
    """
    RNA-seq data generator

    :param n_transcripts: Number of transcripts.
    :param distribution: "Poisson" or "Negative binomial".
    :param n_STA: Number of counts in STA class.
    :param n_CR: Number of counts in CR class (chronic rejection).
    :param n_OT: Number of counts in OT class (operation tolerant).
    :param n_IS_effect: Number of transcripts which are effected by IS.
    :param IS_effect: Min and max beta coefficient (effect of IS).
    :param is_class_effect: True if there is class effect in transcripts.
    :param n_class_effect: Number of transcripts which are effected by class.
    :param class_effect: Mean and std values for normal distribution.
    :param result_file_name: Name of file to save results.
    :return array (n_STA + n_CR + n_OT)
    """

    # coefficients, IS role
    # beta0 = np.log(stats.f.rvs(125.75, 2.99, 1.09, 7.18, size=n_transcripts))
    beta0 = np.log(stats.f.rvs(38141.58, 3.02, -0.22, 5.99, size=n_transcripts))
    # beta0 = np.random.normal(3, 1.2, n_transcripts)
    beta1 = get_random_beta(n_transcripts, n_IS_effect, IS_effect)
    beta2 = get_random_beta(n_transcripts, n_IS_effect, IS_effect)
    beta3 = get_random_beta(n_transcripts, n_IS_effect, IS_effect)

    # 1) Generate RNASeq for one class (the equivalent of STA class)
    counts_STA, expected_counts_STA, IS_STA = generate_rnaseq_class("STA", n_STA, n_transcripts, beta0, beta1, beta2,
                                                                    beta3,
                                                                    distribution=distribution,
                                                                    isClassEffect=False, n_effect=0,
                                                                    class_effect=class_effect,
                                                                    isIS=True)
    RNASeq = counts_STA
    expectedRNASeq = expected_counts_STA
    IS = IS_STA

    # 2) Generate RNASeq for different class (the equivalent of CR class)
    counts_CR, expected_counts_CR, IS_CR = generate_rnaseq_class("CR", n_CR, n_transcripts, beta0, beta1, beta2, beta3,
                                                                 distribution=distribution,
                                                                 isClassEffect=is_class_effect, n_effect=n_class_effect,
                                                                 class_effect=class_effect,
                                                                 isIS=True)
    RNASeq = pd.concat([RNASeq, counts_CR], axis=1)
    expectedRNASeq = pd.concat([expectedRNASeq, expected_counts_CR], axis=1)
    IS = pd.concat([IS, IS_CR], axis=0)

    # 3) Generate RNASeq for different class (the equivalent of OT class)
    counts_OT, expected_counts_OT, IS_OT = generate_rnaseq_class("OT", n_OT, n_transcripts, beta0, beta1, beta2, beta3,
                                                                 distribution=distribution,
                                                                 isClassEffect=is_class_effect, n_effect=n_class_effect,
                                                                 class_effect=class_effect,
                                                                 isIS=False)
    RNASeq = pd.concat([RNASeq, counts_OT], axis=1)
    expectedRNASeq = pd.concat([expectedRNASeq, expected_counts_OT], axis=1)
    IS = pd.concat([IS, IS_OT], axis=0)

    if result_file_name != "":
        RNASeq.to_csv('data/rnaseq_origin_{name}.csv'.format(name=result_file_name))
        expectedRNASeq.to_csv('data/rnaseq_expected_{name}.csv'.format(name=result_file_name))
        IS.to_csv('data/immunosuppression_{name}.csv'.format(name=result_file_name))
    return RNASeq, expectedRNASeq, IS, (counts_STA, counts_CR, counts_OT), (expected_counts_STA, expected_counts_CR, expected_counts_OT), (IS_STA, IS_CR, IS_OT)


def get_random_beta(n_transcripts, n_IS_effect, IS_effect):
    beta = np.append(np.zeros(n_transcripts - n_IS_effect), np.linspace(IS_effect[0], IS_effect[1], n_IS_effect))
    np.random.shuffle(beta)
    return beta


def generate_rnaseq_class(name, n_samples, n_transcripts, beta0, beta1, beta2, beta3, class_effect, isClassEffect,
                          n_effect, isIS, distribution):
    # init immunosuppression
    IS = np.zeros((n_samples, 3))
    if isIS:
        for i in range(3):
            IS[:, i] = np.random.choice([0, 1], p=[0.4, 0.6], size=n_samples, replace=True)

    # effect of class
    if isClassEffect:
        eff = np.random.permutation(np.append(np.zeros(n_transcripts - n_effect),
                                              np.random.normal(class_effect[0], class_effect[1], n_effect)))
        beta0 = beta0 - eff if random.random() > 0.5 else beta0 + eff
        beta0[beta0 < 0] += 2 * eff[beta0 < 0]

    # generate counts with and without IS
    counts = np.zeros((n_transcripts, n_samples), dtype=int)
    counts_without_IS = np.zeros((n_transcripts, n_samples), dtype=int)
    for i in range(n_transcripts):
        mu = np.exp(beta0[i] + beta1[i] * IS[:, 0] + beta2[i] * IS[:, 1] + beta3[i] * IS[:, 2])
        mu_without_IS = np.exp(beta0[i] * np.ones((n_samples,)))
        # Poisson
        if distribution == "Poisson":
            counts[i, :] = np.random.poisson(lam=mu[:n_samples], size=n_samples)
            counts_without_IS[i, :] = np.random.poisson(lam=mu_without_IS[:n_samples], size=n_samples)
        # Negative binomial
        elif distribution == "Negative binomial":
            n, p = estimate_nb_parameters(mu[:n_samples])
            counts[i, :] = np.random.negative_binomial(n=n, p=p, size=n_samples)
            n, p = estimate_nb_parameters(mu_without_IS[:n_samples])
            counts_without_IS[i, :] = np.random.negative_binomial(n=n, p=p, size=n_samples)
        else:
            print('Unknown distribution: ', distribution)

    IS_df = pd.DataFrame(data=IS,
                         columns=["IS1", "IS2", "IS3"],
                         index=[name + str(num) for num in range(n_samples)],
                         dtype=int)

    counts_df = pd.DataFrame(data=counts,
                             index=["transcript" + str(num) for num in range(n_transcripts)],
                             columns=[name + str(num) for num in range(n_samples)],
                             dtype=int)

    counts_without_IS_df = pd.DataFrame(data=counts_without_IS,
                                        index=["transcript" + str(num) for num in range(n_transcripts)],
                                        columns=[name + str(num) for num in range(n_samples)],
                                        dtype=int)

    return counts_df, counts_without_IS_df, IS_df


def estimate_nb_parameters(mean):
    std = np.sqrt(0.04519 * mean ** 2 + 1.17 * mean + 0.1613)
    p = mean / std ** 2
    n = mean * p / (1.0 - p)
    return n, p
