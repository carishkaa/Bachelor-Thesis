import os.path
import sys
sys.path.append("/Users/karinabalagazova/Desktop/cvut/5.semestr/scientificProject/functions/..")
from functions.rnaseq_data_generator import rna_seq_generator, NormalDistributionParameters
from utils.enums import Distribution
import numpy as np

from functions.saving_data import get_empty_auc_dataset
from multiprocessing import Pool

from pathlib import Path
import datetime

import statsmodels.api as sm
import pandas as pd
import random
from functions.classificator import classify
from functions.data_preprocess import preprocess_for_classification


def experiment_classification(n_transcripts, n_IS_effect, IS_effect, n_class_effect, class_effect_mean, STA_range, pool, filename):
    n_repeats = 10 # pocet opakovani
    n_genSTA = 12000 # pocet nagenerovanych STA vzorku
    n_genCR = 100 # pocet nagenerovanych CR vzorku
    n_genOT = 100 # pocet nagenerovanych OT vzorku

    # Generate dataset
    _, _, _, gen_origin, gen_expected, gen_IS = rna_seq_generator(
        n_transcripts=n_transcripts,
        distribution=Distribution.POISSON,
        n_STA=n_genSTA, n_CR=n_genCR, n_OT=n_genOT,
        n_IS_effect=n_IS_effect, IS_effect=(IS_effect, IS_effect+0.01),  # IS effect
        is_class_effect=n_class_effect > 0, n_class_effect=n_class_effect, class_effect=NormalDistributionParameters(class_effect_mean, 0.1)
    )

    # Compensate and classify
    mean_auc_results_origin, mean_auc_results_expected, mean_auc_results_filtered = \
        experiment_iteration(STA_range, n_repeats,
                             gen_origin, gen_expected, gen_IS,
                             n_genSTA, n_genCR, n_genOT, "Poisson",  # "Negative binomial" or "Poisson"
                             pool, filename)
    return mean_auc_results_origin, mean_auc_results_expected, mean_auc_results_filtered


def learn_beta_poisson(df):
    mp = sm.formula.glm("y ~ x1 + x2 + x3", family=sm.families.Poisson(), data=df).fit()
    return np.array([mp.params.x1, mp.params.x2, mp.params.x3])


def learn_beta_neg_binomial(df):
    mp = sm.formula.glm("y ~ x1 + x2 + x3", family=sm.families.NegativeBinomial(), data=df).fit()
    return np.array([mp.params.x1, mp.params.x2, mp.params.x3])


def compensation(counts, IS, t, n_CR, distribution, pool: Pool):
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
    learned_beta = np.zeros((3, t))
    values = list()
    for ind in range(t):
        values.append(pd.DataFrame({"x1": STA_IS['IS1'].values.tolist(),
                                    "x2": STA_IS['IS2'].values.tolist(),
                                    "x3": STA_IS['IS3'].values.tolist(),
                                    "y": STA_counts[ind]}))

    vs = pool.map(learn_beta_poisson, values) if distribution == "Poisson" \
        else pool.map(learn_beta_neg_binomial, values) if distribution == "Negative binomial" \
        else False

    if not vs:
        print('Something is bad with pool.map(...)')

    for ind in range(t):
        learned_beta[:, ind] = vs[ind]

    # leave only CR and OT classes for classification
    OT_CR_counts = counts.loc[:, ~STA_patients].to_numpy(dtype=float)
    OT_CR_IS = IS.loc[~STA_patients].values

    # filter out IS from CR
    OT_CR_counts[:, :n_CR] = OT_CR_counts[:, :n_CR] / np.exp(np.dot(OT_CR_IS[:n_CR], learned_beta).transpose())
    OT_CR_counts = OT_CR_counts.transpose()
    return OT_CR_counts, OT_CR_IS, learned_beta


def experiment(generated_counts, generated_expected_counts, generated_IS, n_CR: int, distribution, pool):
    # counts generated with IS influence
    counts, patients_labels, _ = preprocess_for_classification(generated_counts)

    # counts generated with similar beta0, but without IS influence
    expected_counts, _, _ = preprocess_for_classification(generated_expected_counts)

    # IS compensation
    t = generated_counts.shape[0]
    filtered_counts, OT_CR_IS, learned_beta = compensation(generated_counts, generated_IS, t, n_CR, distribution, pool)

    columns = ["transcript" + str(num) for num in range(counts.shape[1])]
    y = pd.DataFrame(data=patients_labels)

    datasets_to_save = dict()
    datasets_to_save['learned_beta'] = learned_beta
    datasets_to_save['y'] = y

    # COUNTS
    X = pd.DataFrame(data=counts, columns=columns)
    results_origin = classify(X, y.values.ravel(), n_splits=10, n_repeats=1)
    datasets_to_save['origin_X'] = X

    # EXPECTED
    X = pd.DataFrame(data=expected_counts, columns=columns)
    results_expected = classify(X, y.values.ravel(), n_splits=10, n_repeats=1)
    datasets_to_save['expected_X'] = X

    # FILTERED
    X = pd.DataFrame(data=filtered_counts, columns=columns)
    results_filtered = classify(X, y.values.ravel(), n_splits=10, n_repeats=1)
    datasets_to_save['filtered_X'] = X

    return results_origin, results_expected, results_filtered, datasets_to_save


def experiment_iteration(STA_range, n_repeats, gen_origin, gen_expected, gen_IS,
                         n_genSTA, n_genCR, n_genOT,
                         distribution, pool, filename):
    n_STA = STA_range[-1]  # maximal number of STA patients

    # arrays for AUC results
    mean_auc_results_origin = np.zeros((n_repeats, len(STA_range)))
    mean_auc_results_expected = np.zeros((n_repeats, len(STA_range)))
    mean_auc_results_filtered = np.zeros((n_repeats, len(STA_range)))

    for j in range(n_repeats):
        STA_random_index = random.sample(range(n_genSTA), n_STA)
        CR_random_index = random.sample(range(n_genCR), 15)
        OT_random_index = random.sample(range(n_genOT), 15)

        counts_STA = gen_origin[0][gen_origin[0].columns[STA_random_index]]
        counts_CR = gen_origin[1][gen_origin[1].columns[CR_random_index]]
        counts_OT = gen_origin[2][gen_origin[2].columns[OT_random_index]]

        expected_STA = gen_expected[0][gen_expected[0].columns[STA_random_index]]
        expected_CR = gen_expected[1][gen_expected[1].columns[CR_random_index]]
        expected_OT = gen_expected[2][gen_expected[2].columns[OT_random_index]]

        IS_STA = gen_IS[0].loc[gen_IS[0].index[STA_random_index].to_numpy()]
        IS_CR = gen_IS[1].loc[gen_IS[1].index[CR_random_index]]
        IS_OT = gen_IS[2].loc[gen_IS[2].index[OT_random_index]]

        a = pd.concat([counts_STA, counts_CR, counts_OT], axis=1)
        b = pd.concat([expected_STA, expected_CR, expected_OT], axis=1)
        c = pd.concat([IS_STA, IS_CR, IS_OT], axis=0)

        patient_labels = a.columns

        for i in range(len(STA_range)):
            # leave only STA_range[i] STA patients
            a_tmp = a.drop(patient_labels[range(n_STA - STA_range[i])], axis=1)
            b_tmp = b.drop(patient_labels[range(n_STA - STA_range[i])], axis=1)
            c_tmp = c.drop(patient_labels[range(n_STA - STA_range[i])], axis=0)
            try:
                d, e, f, datasets_to_save = experiment(a_tmp, b_tmp, c_tmp, n_CR=15, distribution=distribution, pool=pool)
                mean_auc_results_origin[j, i] = d.mean()
                mean_auc_results_expected[j, i] = e.mean()
                mean_auc_results_filtered[j, i] = f.mean()
                datasets_to_save_filename = '{filename}-STA-{sta}-repeat-{j}-{uniqueID}.npz'.format(filename=filename, sta=STA_range[i], j=j, uniqueID=datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
                np.savez(datasets_to_save_filename, datasets=datasets_to_save)
            except Exception as e:  # when STA number is too small, there is the PerfectSeparation error
                print('Exception when ', STA_range[i], ' STA samples. Error: ', e)
                mean_auc_results_origin[j, i] = np.nan
                mean_auc_results_expected[j, i] = np.nan
                mean_auc_results_filtered[j, i] = np.nan
    return mean_auc_results_origin, mean_auc_results_expected, mean_auc_results_filtered


if __name__ == "__main__":
    data_distribution = 'poisson'  # 'neg-binomial' or 'poisson'

    # Create folder to save generated data
    foldername_generated_data = 'generated-data-' + data_distribution + '-' + datetime.datetime.now().strftime(
        '%Y%m%d-%H%M%S')
    Path(foldername_generated_data).mkdir(parents=True, exist_ok=True)
    print('All datasets will be stored in \"' + foldername_generated_data + '\" folder.')

    # Get and save EMPTY dict for storing classification results
    data = get_empty_auc_dataset()
    filename_auc_results_origin = 'auc_results_origin_' + data_distribution + '.npz'
    filename_auc_results_expected = 'auc_results_expected_' + data_distribution + '.npz'
    filename_auc_results_filtered = 'auc_results_filtered_' + data_distribution + '.npz'

    if os.path.isfile(filename_auc_results_origin) or \
            os.path.isfile(filename_auc_results_expected) or \
            os.path.isfile(filename_auc_results_filtered):
        print('WARNING: File with this name already exists!')
    else:
        np.savez(filename_auc_results_origin, auc_data=data)
        np.savez(filename_auc_results_expected, auc_data=data)
        np.savez(filename_auc_results_filtered, auc_data=data)

    # Arrays for storing classification results
    saved_origin = np.load(filename_auc_results_origin, allow_pickle=True)
    saved_origin = saved_origin['auc_data'].item()

    saved_expected = np.load(filename_auc_results_expected, allow_pickle=True)
    saved_expected = saved_expected['auc_data'].item()

    saved_filtered = np.load(filename_auc_results_filtered, allow_pickle=True)
    saved_filtered = saved_filtered['auc_data'].item()

    # EXPERIMENTS
    transcripts = [10000]
    class_effect_n_transcripts = [0, 5, 10, 20]
    class_effect = [0.3, 0.4]  # mean of normal distribution

    IS_effect_n_transcripts = [5, 10]
    IS_effect = [0.5, 0.9]  # coefficients beta1..beta3

    STA_range = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    with Pool() as pool:
        for t in transcripts:
            for n_IS in IS_effect_n_transcripts:
                print('new IS_effect_n_transcripts: ', n_IS)
                for i in IS_effect:
                    print('new IS_effect: ', i)
                    for n_class in class_effect_n_transcripts:
                        if n_class == 0:
                            c = 0.2
                            filename = foldername_generated_data + '/{t}_{n_IS}_{i}_{n_class}_{dist}'.format(t=t,
                                                                                                             n_IS=n_IS,
                                                                                                             i=i,
                                                                                                             n_class=n_class,
                                                                                                             dist=data_distribution)
                            mean_o, mean_e, mean_f = experiment_classification(t, n_IS, i, n_class, c, STA_range, pool,
                                                                               filename)
                            saved_origin[t][n_IS][i][n_class][c]['STA_range'] = STA_range
                            saved_origin[t][n_IS][i][n_class][c]['auc'] = mean_o

                            saved_expected[t][n_IS][i][n_class][c]['STA_range'] = STA_range
                            saved_expected[t][n_IS][i][n_class][c]['auc'] = mean_e

                            saved_filtered[t][n_IS][i][n_class][c]['STA_range'] = STA_range
                            saved_filtered[t][n_IS][i][n_class][c]['auc'] = mean_f
                        else:
                            for c in class_effect:
                                filename = foldername_generated_data + '/{t}_{n_IS}_{i}_{n_class}_{c}_{dist}'.format(
                                    t=t, n_IS=n_IS, i=i, n_class=n_class, c=c, dist=data_distribution)

                                if (len(saved_origin[t][n_IS][i][n_class][c]['STA_range']) > 0 and
                                        saved_origin[t][n_IS][i][n_class][c]['auc'].size > 0):
                                    print('WARNING: Dataset with parameters \"' + filename + '...\" already exists')
                                    continue

                                mean_o, mean_e, mean_f = experiment_classification(t, n_IS, i, n_class, c, STA_range,
                                                                                   pool, filename)
                                saved_origin[t][n_IS][i][n_class][c]['STA_range'] = STA_range
                                saved_origin[t][n_IS][i][n_class][c]['auc'] = mean_o

                                saved_expected[t][n_IS][i][n_class][c]['STA_range'] = STA_range
                                saved_expected[t][n_IS][i][n_class][c]['auc'] = mean_e

                                saved_filtered[t][n_IS][i][n_class][c]['STA_range'] = STA_range
                                saved_filtered[t][n_IS][i][n_class][c]['auc'] = mean_f

                        np.savez(filename_auc_results_origin, auc_data=saved_origin)
                        np.savez(filename_auc_results_expected, auc_data=saved_expected)
                        np.savez(filename_auc_results_filtered, auc_data=saved_filtered)
