def precision_S(y_test, y_pred):
    corr_negative, corr_issue, corr_solution = 0, 0, 0
    count_negative, count_issue, count_solution = 0, 0, 0
    for i, prediction in enumerate(y_pred):
        if prediction[0] == 1:
            count_negative += 1
            if y_test[i][0] == 1:
                corr_negative += 1
        if prediction[1] == 1:
            count_issue += 1
            if y_test[i][1] == 1:
                corr_issue += 1
        # if prediction[2] == 1:
        #     count_solution += 1
        #     if y_test[i][2] == 1:
        #         corr_solution += 1
    precision_negative = float(corr_negative) / float(count_negative)
    precision_issue = float(corr_issue) / float(count_issue)
    # precision_solution = float(corr_solution) / float(count_solution)

    return precision_issue


def recall_S(y_test, y_pred):
    # truth_number = 0
    # corr_number = 0
    corr_negative, corr_issue, corr_solution = 0, 0, 0
    count_negative, count_issue, count_solution = 0, 0, 0
    for i, testing in enumerate(y_test):
        if testing[0] == 1:
            count_negative += 1
            if y_pred[i][0] == 1:
                corr_negative += 1
        if testing[1] == 1:
            count_issue += 1
            if y_pred[i][1] == 1:
                corr_issue += 1
        # if testing[2] == 1:
        #     count_solution += 1
        #     if y_pred[i][2] == 1:
        #         corr_solution += 1
        # test_pos_list = []
        # for j in range(3):
        #     if testing[j] == 1:
        #         test_pos_list.append(j)
        # if 0 not in test_pos_list and (1 in test_pos_list or 2 in test_pos_list):
        #     truth_number += 1
        #     for truth_index in test_pos_list:
        #         if y_pred[i][truth_index] == 1 and truth_index != 0:
        #             corr_number += 1
        #             break
    recall_negative = float(corr_negative) / float(count_negative)
    recall_issue = float(corr_issue) / float(count_issue)
    # recall_solution = float(corr_solution) / float(count_solution)

    return recall_issue