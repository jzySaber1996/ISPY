import numpy as np
import random

def cross_project_selection(train_projects, valid_projects, test_projects, filter_issue=False):
    train_file = '../data/issuedialog/train.tsv'
    valid_file = '../data/issuedialog/valid.tsv'
    test_file = '../data/issuedialog/test.tsv'

    train_features = '../data/issuedialog/train_features.tsv'
    valid_features = '../data/issuedialog/valid_features.tsv'
    test_features = '../data/issuedialog/test_features.tsv'
    with open(train_file, 'w', encoding='utf8') as train_out, open(valid_file, 'w', encoding='utf8') as valid_out, \
            open(test_file, 'w', encoding='utf8') as test_out:
        with open(train_features, 'w', encoding='utf8') as train_feat_out, \
                open(valid_features, 'w', encoding='utf8') as valid_feat_out, \
                open(test_features, 'w', encoding='utf8') as test_feat_out:
            # train-valid split
            issue_storage, non_issue_storage, issue_feat_storage, non_issue_feat_storage = [], [], [], []
            for train_project in train_projects:
                with open(f'../data/issuedialog/{train_project}.tsv', encoding='utf8') as train_in,\
                        open(f'../data/issuedialog/{train_project}_features.tsv', encoding='utf8') as train_feat_in:
                    data_projects = train_in.readlines()
                    data_feat_projects = train_feat_in.readlines()
                    count_within_issue, count_answer = 0, 0
                    each_issue, each_non_issue, each_issue_feat, each_non_issue_feat = [], [], [], []
                    for data_project, data_feat_project in zip(data_projects, data_feat_projects):
                        if not filter_issue:
                            train_out.write(data_project)
                            train_feat_out.write(data_feat_project)
                        else:
                            data_project_elements = data_project.split('\t')
                            if len(data_project_elements) == 5 and data_project_elements[4] == '1\n':
                                count_within_issue = 1
                                # if count_within_issue == 1:
                                #     train_out.write('\n')
                                #     train_feat_out.write('\n')
                                each_issue.append([data_project, data_feat_project])
                                each_issue_feat.append(data_feat_project)
                                # train_out.write(data_project)
                                # train_feat_out.write(data_feat_project)
                            elif len(data_project_elements) == 5 and data_project_elements[4] == '0\n':
                                count_within_issue = 0
                                if data_project_elements[3] == '1':
                                    count_answer = 1
                                each_non_issue.append([data_project, data_feat_project])
                                each_non_issue_feat.append(data_feat_project)
                            elif data_project == '\n':
                                if count_within_issue == 1:
                                    issue_storage.append(each_issue)
                                    issue_feat_storage.append(each_issue_feat)
                                elif count_within_issue == 0 and count_answer == 0:
                                    non_issue_storage.append(each_non_issue)
                                    non_issue_feat_storage.append(each_non_issue_feat)
                                each_issue, each_non_issue, each_issue_feat, each_non_issue_feat = [], [], [], []
                                count_answer = 0
                train_in.close()
                train_feat_in.close()
            # np_issue, np_issue_feat = np.array(issue_storage), np.array(issue_feat_storage)
            random.shuffle(non_issue_storage)
            non_issue_storage = non_issue_storage[0: len(issue_storage)]
            total_train_dataset = issue_storage + non_issue_storage
            random.shuffle(total_train_dataset)
            train_set, valid_set = total_train_dataset[0: int(9 * len(total_train_dataset)/10)],\
                                   total_train_dataset[int(9 * len(total_train_dataset)/10):]
            for train_data in train_set:
                for each_train_sentence in train_data:
                    train_out.write(each_train_sentence[0])
                    train_feat_out.write(each_train_sentence[1])
                train_out.write('\n')
                train_feat_out.write('\n')
            for valid_data in valid_set:
                for each_valid_sentence in valid_data:
                    valid_out.write(each_valid_sentence[0])
                    valid_feat_out.write(each_valid_sentence[1])
                valid_out.write('\n')
                valid_feat_out.write('\n')
            train_out.close()
            train_feat_out.close()
            valid_out.close()
            valid_feat_out.close()

            # cross-project test split
            issue_storage, non_issue_storage, issue_feat_storage, non_issue_feat_storage = [], [], [], []
            for test_project in test_projects:
                with open(f'../data/issuedialog/{test_project}.tsv', encoding='utf8') as test_in,\
                        open(f'../data/issuedialog/{test_project}_features.tsv', encoding='utf8') as test_feat_in:
                    data_projects = test_in.readlines()
                    data_feat_projects = test_feat_in.readlines()
                    count_within_issue, count_answer = 0, 0
                    each_issue, each_non_issue, each_issue_feat, each_non_issue_feat = [], [], [], []
                    for data_project, data_feat_project in zip(data_projects, data_feat_projects):
                        if not filter_issue:
                            test_out.write(data_project)
                            test_feat_out.write(data_feat_project)
                        else:
                            data_project_elements = data_project.split('\t')
                            if len(data_project_elements) == 5 and data_project_elements[4] == '1\n':
                                count_within_issue = 1
                                # if count_within_issue == 1:
                                #     train_out.write('\n')
                                #     train_feat_out.write('\n')
                                each_issue.append([data_project, data_feat_project])
                                each_issue_feat.append(data_feat_project)
                                # train_out.write(data_project)
                                # train_feat_out.write(data_feat_project)
                            elif len(data_project_elements) == 5 and data_project_elements[4] == '0\n':
                                count_within_issue = 0
                                if data_project_elements[3] == '1':
                                    count_answer = 1
                                each_non_issue.append([data_project, data_feat_project])
                                each_non_issue_feat.append(data_feat_project)
                            elif data_project == '\n':
                                if count_within_issue == 1:
                                    issue_storage.append(each_issue)
                                    issue_feat_storage.append(each_issue_feat)
                                elif count_within_issue == 0 and count_answer == 0:
                                    non_issue_storage.append(each_non_issue)
                                    non_issue_feat_storage.append(each_non_issue_feat)
                                each_issue, each_non_issue, each_issue_feat, each_non_issue_feat = [], [], [], []
                                count_answer = 0
                test_in.close()
                test_feat_in.close()
                # np_issue, np_issue_feat = np.array(issue_storage), np.array(issue_feat_storage)
            random.shuffle(non_issue_storage)
            non_issue_storage = non_issue_storage[0: len(issue_storage)]
            total_test_dataset = issue_storage + non_issue_storage
            random.shuffle(total_test_dataset)
            for test_data in total_test_dataset:
                for each_test_sentence in test_data:
                    test_out.write(each_test_sentence[0])
                    test_feat_out.write(each_test_sentence[1])
                test_out.write('\n')
                test_feat_out.write('\n')
            test_out.close()
            test_feat_out.close()
            print('Data Prepare Finished')
            # for valid_project in valid_projects:
            #     # with open(f'../data/issuedialog/{valid_project}.tsv', encoding='utf8') as valid_in:
            #     #     data_projects = valid_in.readlines()
            #     #     for data_project in data_projects:
            #     #         valid_out.write(data_project)
            #     # valid_in.close()
            #     # with open(f'../data/issuedialog/{valid_project}_features.tsv', encoding='utf8') as valid_in:
            #     #     data_projects = valid_in.readlines()
            #     #     for data_project in data_projects:
            #     #         valid_feat_out.write(data_project)
            #     # valid_in.close()
            #     with open(f'../data/issuedialog/{valid_project}.tsv', encoding='utf8') as valid_in,\
            #             open(f'../data/issuedialog/{valid_project}_features.tsv', encoding='utf8') as valid_feat_in:
            #         data_projects = valid_in.readlines()
            #         data_feat_projects = valid_feat_in.readlines()
            #         count_within_issue = 0
            #         for data_project, data_feat_project in zip(data_projects, data_feat_projects):
            #             if not filter_issue:
            #                 valid_out.write(data_project)
            #                 valid_feat_out.write(data_feat_project)
            #             else:
            #                 data_project_elements = data_project.split('\t')
            #                 if len(data_project_elements) == 5 and data_project_elements[4] == '1\n':
            #                     count_within_issue += 1
            #                     if count_within_issue == 1:
            #                         valid_out.write('\n')
            #                         valid_feat_out.write('\n')
            #                     valid_out.write(data_project)
            #                     valid_feat_out.write(data_feat_project)
            #                 else:
            #                     count_within_issue = 0
            #     valid_in.close()
            #     valid_feat_in.close()
    return




def cross_project_selection_origin(train_projects, test_projects, filter_issue=False):
    train_file = f'../data/issuedialog_projects/train_origin_{test_projects[0]}.tsv'
    valid_file = f'../data/issuedialog_projects/valid_origin_{test_projects[0]}.tsv'
    test_file = f'../data/issuedialog_projects/test_origin_{test_projects[0]}.tsv'

    train_features = f'../data/issuedialog_projects/train_features_origin_{test_projects[0]}.tsv'
    valid_features = f'../data/issuedialog_projects/valid_features_origin_{test_projects[0]}.tsv'
    test_features = f'../data/issuedialog_projects/test_features_origin_{test_projects[0]}.tsv'
    with open(train_file, 'w', encoding='utf8') as train_out, open(valid_file, 'w', encoding='utf8') as valid_out, \
            open(test_file, 'w', encoding='utf8') as test_out:
        with open(train_features, 'w', encoding='utf8') as train_feat_out, \
                open(valid_features, 'w', encoding='utf8') as valid_feat_out, \
                open(test_features, 'w', encoding='utf8') as test_feat_out:
            # train-valid split
            issue_storage, non_issue_storage, issue_feat_storage, non_issue_feat_storage = [], [], [], []
            for train_project in train_projects:
                with open(f'../data/issuedialog/{train_project}.tsv', encoding='utf8') as train_in,\
                        open(f'../data/issuedialog/{train_project}_features.tsv', encoding='utf8') as train_feat_in:
                    data_projects = train_in.readlines()
                    data_feat_projects = train_feat_in.readlines()
                    count_within_issue, count_answer = 0, 0
                    each_issue, each_non_issue, each_issue_feat, each_non_issue_feat = [], [], [], []
                    for data_project, data_feat_project in zip(data_projects, data_feat_projects):
                        if not filter_issue:
                            train_out.write(data_project)
                            train_feat_out.write(data_feat_project)
                        else:
                            data_project_elements = data_project.split('\t')
                            if len(data_project_elements) == 5 and data_project_elements[4] == '1\n':
                                count_within_issue = 1
                                # if count_within_issue == 1:
                                #     train_out.write('\n')
                                #     train_feat_out.write('\n')
                                each_issue.append([data_project, data_feat_project])
                                each_issue_feat.append(data_feat_project)
                                # train_out.write(data_project)
                                # train_feat_out.write(data_feat_project)
                            elif len(data_project_elements) == 5 and data_project_elements[4] == '0\n':
                                count_within_issue = 0
                                if data_project_elements[3] == '1':
                                    count_answer = 1
                                each_non_issue.append([data_project, data_feat_project])
                                each_non_issue_feat.append(data_feat_project)
                            elif data_project == '\n':
                                if count_within_issue == 1:
                                    issue_storage.append(each_issue)
                                    issue_feat_storage.append(each_issue_feat)
                                elif count_within_issue == 0:
                                    non_issue_storage.append(each_non_issue)
                                    non_issue_feat_storage.append(each_non_issue_feat)
                                each_issue, each_non_issue, each_issue_feat, each_non_issue_feat = [], [], [], []
                                count_answer = 0
                train_in.close()
                train_feat_in.close()
            # np_issue, np_issue_feat = np.array(issue_storage), np.array(issue_feat_storage)
            random.shuffle(non_issue_storage)
            total_train_dataset = issue_storage + non_issue_storage
            random.shuffle(total_train_dataset)
            train_set, valid_set = total_train_dataset[0: int(9 * len(total_train_dataset)/10)],\
                                   total_train_dataset[int(9 * len(total_train_dataset)/10):]
            for train_data in train_set:
                for each_train_sentence in train_data:
                    train_out.write(each_train_sentence[0])
                    train_feat_out.write(each_train_sentence[1])
                train_out.write('\n')
                train_feat_out.write('\n')
            for valid_data in valid_set:
                for each_valid_sentence in valid_data:
                    valid_out.write(each_valid_sentence[0])
                    valid_feat_out.write(each_valid_sentence[1])
                valid_out.write('\n')
                valid_feat_out.write('\n')
            train_out.close()
            train_feat_out.close()
            valid_out.close()
            valid_feat_out.close()

            # cross-project test split
            issue_storage, non_issue_storage, issue_feat_storage, non_issue_feat_storage = [], [], [], []
            for test_project in test_projects:
                with open(f'../data/issuedialog/{test_project}.tsv', encoding='utf8') as test_in,\
                        open(f'../data/issuedialog/{test_project}_features.tsv', encoding='utf8') as test_feat_in:
                    data_projects = test_in.readlines()
                    data_feat_projects = test_feat_in.readlines()
                    count_within_issue, count_answer = 0, 0
                    each_issue, each_non_issue, each_issue_feat, each_non_issue_feat = [], [], [], []
                    for data_project, data_feat_project in zip(data_projects, data_feat_projects):
                        if not filter_issue:
                            test_out.write(data_project)
                            test_feat_out.write(data_feat_project)
                        else:
                            data_project_elements = data_project.split('\t')
                            if len(data_project_elements) == 5 and data_project_elements[4] == '1\n':
                                count_within_issue = 1
                                # if count_within_issue == 1:
                                #     train_out.write('\n')
                                #     train_feat_out.write('\n')
                                each_issue.append([data_project, data_feat_project])
                                each_issue_feat.append(data_feat_project)
                                # train_out.write(data_project)
                                # train_feat_out.write(data_feat_project)
                            elif len(data_project_elements) == 5 and data_project_elements[4] == '0\n':
                                count_within_issue = 0
                                if data_project_elements[3] == '1':
                                    count_answer = 1
                                each_non_issue.append([data_project, data_feat_project])
                                each_non_issue_feat.append(data_feat_project)
                            elif data_project == '\n':
                                if count_within_issue == 1:
                                    issue_storage.append(each_issue)
                                    issue_feat_storage.append(each_issue_feat)
                                elif count_within_issue == 0:
                                    non_issue_storage.append(each_non_issue)
                                    non_issue_feat_storage.append(each_non_issue_feat)
                                each_issue, each_non_issue, each_issue_feat, each_non_issue_feat = [], [], [], []
                                count_answer = 0
                test_in.close()
                test_feat_in.close()
                # np_issue, np_issue_feat = np.array(issue_storage), np.array(issue_feat_storage)
            random.shuffle(non_issue_storage)
            total_test_dataset = issue_storage + non_issue_storage
            random.shuffle(total_test_dataset)
            for test_data in total_test_dataset:
                for each_test_sentence in test_data:
                    test_out.write(each_test_sentence[0])
                    test_feat_out.write(each_test_sentence[1])
                test_out.write('\n')
                test_feat_out.write('\n')
            test_out.close()
            test_feat_out.close()
            print('Data Prepare Finished')

if __name__=='__main__':
    projects = ['Angular', 'Appium', 'Deeplearning4j', 'Docker', 'Ethereum', 'Nodejs', 'Gitter', 'Typescript']
    # train_projects, test_projects, valid_projects = projects[0: 2] + projects[3:], [projects[2]], [projects[7]]
    # # cross_project_selection(train_projects, valid_projects, test_projects, filter_issue=True)
    # cross_project_selection_origin(train_projects, test_projects, filter_issue=True)
    for i in range(8):
        train_projects, test_projects, valid_projects = projects[0: i] + projects[i + 1:], [projects[i]], [projects[7]]
        cross_project_selection_origin(train_projects, test_projects, filter_issue=True)
