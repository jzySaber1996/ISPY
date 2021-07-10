import xlrd
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


def draw_ablation():
    workbook = xlrd.open_workbook('../data/result_data_new.xlsx')
    sheet = workbook.sheet_by_name('Ablation_study')
    local_attn_data = sheet.col_values(2, 1, sheet.nrows)
    heu_data = sheet.col_values(3, 1, sheet.nrows)
    cnn_data = sheet.col_values(4, 1, sheet.nrows)
    richa_data = sheet.col_values(5, 1, sheet.nrows)

    pre_issue = [[attn, heu, cnn, richa] for i, (attn, heu, cnn, richa)
                 in enumerate(zip(local_attn_data, heu_data, cnn_data, richa_data)) if (i + 1) % 6 == 1]
    rec_issue = [[attn, heu, cnn, richa] for i, (attn, heu, cnn, richa)
                 in enumerate(zip(local_attn_data, heu_data, cnn_data, richa_data)) if (i + 1) % 6 == 2]
    f1_issue = [[attn, heu, cnn, richa] for i, (attn, heu, cnn, richa)
                 in enumerate(zip(local_attn_data, heu_data, cnn_data, richa_data)) if (i + 1) % 6 == 3]
    pre_solution = [[attn, heu, cnn, richa] for i, (attn, heu, cnn, richa)
                 in enumerate(zip(local_attn_data, heu_data, cnn_data, richa_data)) if (i + 1) % 6 == 4]
    rec_solution = [[attn, heu, cnn, richa] for i, (attn, heu, cnn, richa)
                 in enumerate(zip(local_attn_data, heu_data, cnn_data, richa_data)) if (i + 1) % 6 == 5]
    f1_solution = [[attn, heu, cnn, richa] for i, (attn, heu, cnn, richa)
                 in enumerate(zip(local_attn_data, heu_data, cnn_data, richa_data)) if (i + 1) % 6 == 0]
    df_pre_issue = pd.DataFrame({'richa_localattn': [data[0] for data in pre_issue],
                           'richa_heu': [data[1] for data in pre_issue],
                           'richa_cnn': [data[2] for data in pre_issue],
                           'richa': [data[3] for data in pre_issue]})
    df_rec_issue = pd.DataFrame({'richa_localattn': [data[0] for data in rec_issue],
                           'richa_heu': [data[1] for data in rec_issue],
                           'richa_cnn': [data[2] for data in rec_issue],
                           'richa': [data[3] for data in rec_issue]})
    df_f1_issue = pd.DataFrame({'richa_localattn': [data[0] for data in f1_issue],
                           'richa_heu': [data[1] for data in f1_issue],
                           'richa_cnn': [data[2] for data in f1_issue],
                           'richa': [data[3] for data in f1_issue]})
    df_pre_solution = pd.DataFrame({'richa_localattn': [data[0] for data in pre_solution],
                           'richa_heu': [data[1] for data in pre_solution],
                           'richa_cnn': [data[2] for data in pre_solution],
                           'richa': [data[3] for data in pre_solution]})
    df_rec_solution = pd.DataFrame({'richa_localattn': [data[0] for data in rec_solution],
                           'richa_heu': [data[1] for data in rec_solution],
                           'richa_cnn': [data[2] for data in rec_solution],
                           'richa': [data[3] for data in rec_solution]})
    df_f1_solution = pd.DataFrame({'richa_localattn': [data[0] for data in f1_solution],
                           'richa_heu': [data[1] for data in f1_solution],
                           'richa_cnn': [data[2] for data in f1_solution],
                           'richa': [data[3] for data in f1_solution]})
    x_data = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
    # plt.plot(x_data, df_pre_issue.richa)
    # plt.plot(x_data, df_pre_issue.richa_localattn)
    fig = plt.figure()
    plt.subplot(231)
    plt.plot(x_data, list(df_pre_issue.richa), color='limegreen', linestyle='-', marker='s', markersize=4,
             mfcalt='b', label='ISPY')
    plt.xticks([])
    plt.plot(x_data, list(df_pre_issue.richa_localattn), color='darksalmon', linestyle='-', marker='x', markersize=4,
             mfcalt='b', label='ISPY-LocalAttn')
    plt.plot(x_data, list(df_pre_issue.richa_heu), color='orangered', linestyle='-', marker='^', markersize=4,
             mfcalt='b', label='ISPY-Heu')
    plt.plot(x_data, list(df_pre_issue.richa_cnn), color='deepskyblue', linestyle='-', marker='.', mfc='w',
             markersize=4, mfcalt='b', label='ISPY-CNN')
    # plt.grid(axis='y', linestyle='-.')
    # plt.grid(axis='x', linestyle='-.')

    plt.ylabel('Issue-P', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylim([0, 1])
    plt.yticks(fontproperties='Times New Roman', size=13)
    plt.xticks(fontproperties='Times New Roman', size=13)
    # print(stats.ttest_ind(df_pre_issue.richa_heu, df_pre_issue.richa_cnn))

    plt.subplot(232)
    plt.plot(x_data, list(df_rec_issue.richa), color='limegreen', linestyle='-', marker='s', markersize=4,
             mfcalt='b')
    plt.plot(x_data, list(df_rec_issue.richa_localattn), color='darksalmon', linestyle='-', marker='x', markersize=4,
             mfcalt='b')
    plt.plot(x_data, list(df_rec_issue.richa_heu), color='orangered', linestyle='-', marker='^', markersize=4,
             mfcalt='b')
    plt.plot(x_data, list(df_rec_issue.richa_cnn), color='deepskyblue', linestyle='-', marker='.', mfc='w',
             markersize=4, mfcalt='b')
    plt.xticks([])
    plt.yticks([])


    # plt.grid(axis='y', linestyle='-.')
    # plt.grid(axis='x', linestyle='-.')

    plt.ylabel('Issue-R', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylim([0, 1])
    plt.yticks(fontproperties='Times New Roman', size=13)
    plt.xticks(fontproperties='Times New Roman', size=13)
    # print(stats.ttest_ind(df_rec_issue.richa, df_rec_issue.richa_cnn))
    # print(stats.ttest_ind(df_rec_issue.richa, df_rec_issue.richa_localattn))
    # print(stats.ttest_ind(df_rec_issue.richa_heu, df_rec_issue.richa_cnn))


    plt.subplot(233)
    plt.plot(x_data, list(df_f1_issue.richa), color='limegreen', linestyle='-', marker='s', markersize=4,
             mfcalt='b')
    plt.plot(x_data, list(df_f1_issue.richa_localattn), color='darksalmon', linestyle='-', marker='x', markersize=4,
             mfcalt='b')
    plt.plot(x_data, list(df_f1_issue.richa_heu), color='orangered', linestyle='-', marker='^', markersize=4,
             mfcalt='b')
    plt.plot(x_data, list(df_f1_issue.richa_cnn), color='deepskyblue', linestyle='-', marker='.', mfc='w',
             markersize=4, mfcalt='b')
    plt.xticks([])
    plt.yticks([])


    # plt.grid(axis='y', linestyle='-.')
    # plt.grid(axis='x', linestyle='-.')

    plt.ylabel('Issue-F1', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylim([0, 1])
    plt.yticks(fontproperties='Times New Roman', size=13)
    plt.xticks(fontproperties='Times New Roman', size=13)
    print(stats.ttest_ind(df_f1_issue.richa, df_f1_issue.richa_cnn))
    print(stats.ttest_ind(df_f1_issue.richa, df_f1_issue.richa_heu))


    plt.subplot(234)
    plt.plot(x_data, list(df_pre_solution.richa), color='limegreen', linestyle='-', marker='s', markersize=4,
             mfcalt='b')
    plt.plot(x_data, list(df_pre_solution.richa_localattn), color='darksalmon', linestyle='-', marker='x', markersize=4,
             mfcalt='b')
    plt.plot(x_data, list(df_pre_solution.richa_heu), color='orangered', linestyle='-', marker='^', markersize=4,
             mfcalt='b')
    plt.plot(x_data, list(df_pre_solution.richa_cnn), color='deepskyblue', linestyle='-', marker='.', mfc='w',
             markersize=4, mfcalt='b')
    # plt.grid(axis='y', linestyle='-.')
    # plt.grid(axis='x', linestyle='-.')

    plt.ylabel('Solution-P', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylim([0, 1])
    plt.yticks(fontproperties='Times New Roman', size=13)
    plt.xticks(fontproperties='Times New Roman', size=13)
    # print(stats.ttest_ind(df_pre_solution.richa_heu, df_pre_solution.richa_cnn))


    plt.subplot(235)
    plt.plot(x_data, list(df_rec_solution.richa), color='limegreen', linestyle='-', marker='s', markersize=4,
             mfcalt='b')
    plt.plot(x_data, list(df_rec_solution.richa_localattn), color='darksalmon', linestyle='-', marker='x', markersize=4,
             mfcalt='b')
    plt.plot(x_data, list(df_rec_solution.richa_heu), color='orangered', linestyle='-', marker='^', markersize=4,
             mfcalt='b')
    plt.plot(x_data, list(df_rec_solution.richa_cnn), color='deepskyblue', linestyle='-', marker='.', mfc='w',
             markersize=4, mfcalt='b')
    # plt.grid(axis='y', linestyle='-.')
    # plt.grid(axis='x', linestyle='-.')
    plt.yticks([])


    plt.ylabel('Solution-R', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylim([0, 1])
    plt.yticks(fontproperties='Times New Roman', size=13)
    plt.xticks(fontproperties='Times New Roman', size=13)
    # print(stats.ttest_ind(df_rec_solution.richa_heu, df_rec_solution.richa_cnn))

    plt.subplot(236)
    plt.plot(x_data, list(df_f1_solution.richa), color='limegreen', linestyle='-', marker='s', markersize=4,
             mfcalt='b')
    plt.plot(x_data, list(df_f1_solution.richa_localattn), color='darksalmon', linestyle='-', marker='x', markersize=4,
             mfcalt='b')
    plt.plot(x_data, list(df_f1_solution.richa_heu), color='orangered', linestyle='-', marker='^', markersize=4,
             mfcalt='b')
    plt.plot(x_data, list(df_f1_solution.richa_cnn), color='deepskyblue', linestyle='-', marker='.', mfc='w',
             markersize=4, mfcalt='b')
    # plt.grid(axis='y', linestyle='-.')
    # plt.grid(axis='x', linestyle='-.')
    plt.yticks([])


    plt.ylabel('Solution-F1', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylim([0, 1])
    plt.yticks(fontproperties='Times New Roman', size=13)
    plt.xticks(fontproperties='Times New Roman', size=13)
    print(stats.ttest_ind(df_f1_solution.richa, df_f1_solution.richa_cnn))
    print(stats.ttest_ind(df_f1_solution.richa, df_f1_solution.richa_heu))
    fig.legend(loc='upper center', ncol=4, prop={'size': 13, 'family': 'Times New Roman'})
    plt.show()
    # print(df_pre)


def t_return():
    richa = [0.76, 0.77, 0.76, 0.75, 0.68, 0.71, 0.84, 0.74, 0.79, 0.77, 0.68, 0.72, 0.82, 0.73, 0.77, 0.80, 0.69, 0.74, 0.79, 0.70, 0.74, 0.86, 0.78, 0.82]
    nb = [0.36, 0.40, 0.38, 0.41, 0.30, 0.35, 0.47, 0.36, 0.41, 0.70, 0.56, 0.62, 0.08, 0.25, 0.13, 0.22, 0.42, 0.29, 0.30, 0.50, 0.37, 0.15, 0.40, 0.22]
    rf = [0.56, 0.25, 0.34, 0.69, 0.30, 0.42, 0.75, 0.23, 0.35, 0.84, 0.44, 0.58, 1.00, 0.17, 0.29, 0.50, 0.25, 0.33, 0.33, 0.13, 0.18, 0.23, 0.30, 0.26]
    gdbt = [0.27, 0.75, 0.40, 0.40, 0.70, 0.51, 0.50, 0.79, 0.61, 0.73, 0.44, 0.55, 0.21, 0.76, 0.33, 0.19, 0.67, 0.29, 0.30, 0.88, 0.44, 0.18, 0.90, 0.30]
    casper = [0.39, 0.35, 0.37, 0.08, 0.03, 0.05, 0.59, 0.26, 0.36, 0.46, 0.40, 0.43, 0.19, 0.42, 0.26, 0.14, 0.17, 0.15, 0.05, 0.06, 0.06, 0.15, 0.40, 0.22]
    cnc = [0.20, 0.55, 0.29, 0.23, 0.50, 0.32, 0.23, 0.36, 0.28, 0.12, 0.32, 0.17, 0.24, 0.42, 0.30, 0.12, 0.42, 0.19, 0.10, 0.50, 0.17, 0.05, 0.40, 0.10]
    deca = [0.33, 0.50, 0.40, 0.28, 0.37, 0.31, 0.33, 0.36, 0.34, 0.64, 0.28, 0.39, 0.42, 0.42, 0.42, 0.44, 0.67, 0.53, 0.32, 0.50, 0.39, 0.04, 0.10, 0.06]

    baselines = {'nb': nb, 'rf': rf, 'gdbt': gdbt, 'casper': casper, 'cnc': cnc, 'deca': deca}
    for baseline in baselines.keys():
        data_temp = baselines[baseline]
        richa_pre = [ric_value for i, ric_value in enumerate(richa) if (i + 1) % 3 == 1]
        richa_rec = [ric_value for i, ric_value in enumerate(richa) if (i + 1) % 3 == 2]
        richa_f1 = [ric_value for i, ric_value in enumerate(richa) if (i + 1) % 3 == 0]

        base_pre = [base_value for i, base_value in enumerate(data_temp) if (i + 1) % 3 == 1]
        base_rec = [base_value for i, base_value in enumerate(data_temp) if (i + 1) % 3 == 2]
        base_f1 = [base_value for i, base_value in enumerate(data_temp) if (i + 1) % 3 == 0]
        data_t = stats.ttest_ind(richa_f1, base_f1)
        print(data_t)


if __name__=='__main__':
    # t_return()
    draw_ablation()