import os


def read_dialogs():
    path = '../proposed_dataset/new_project'
    output_path = '../proposed_dataset/new_test_cross_project/'
    candidate_files = []
    for root, dirs, files in os.walk(path):
        print("root", root)
        print("dirs", dirs)
        print("files", files)
        candidate_files = files
    for dialog_file in candidate_files:
        dialog_list, dialog = [], []
        with open(path + '/' + dialog_file, mode='r', encoding='utf8') as fin:
            utterance_list = fin.readlines()
        with open(output_path + dialog_file.replace('.txt', '') + '.tsv', 'w', encoding='utf8') as fout:
            user_name = ''
            for utterance in utterance_list:
                if '-------------------------------------' in utterance:
                    # dialog_list.append(dialog)
                    start_utterance = ''
                    count_dialog = 0
                    for i, dialog_each in enumerate(dialog):
                        if dialog_each[0] == 'User':
                            start_utterance += (dialog_each[1] + ' ')
                            count_dialog += 1
                        else:
                            break
                    fout.write('TP' + '\t' + start_utterance.replace('\n', '').replace('\t', ' ') + '__eou__' + '\t' + 'User' + '\n')
                    for i, dialog_utter in enumerate(dialog):
                        if i >= count_dialog:
                            fout.write(
                                'TP' + '\t' + dialog_utter[1].replace('\n', '').replace('\t', ' ') + '__eou__' + '\t' + dialog_utter[0] + '\n')
                    fout.write('\n')
                    dialog = []
                else:
                    if (utterance[0].isdigit() or utterance[0] == '[') and '<' in utterance and '>' in utterance:
                        utterance_user = utterance[utterance.index('<') + 1: utterance.index('>')]
                        utterance_text = utterance[utterance.index('>') + 2:]
                        if len(dialog) == 0:
                            user_name = utterance_user
                        if utterance_user == user_name:
                            dialog.append(['User', utterance_text])
                        else:
                            dialog.append(['Agent', utterance_text])
                        # fout.write('TP' + '\t' + )
            fout.close()
        # print(dialog_list)
    return


def calculate_dialogs():
    path = '../proposed_dataset/test_cross_project/angular.tsv'
    with open(path, mode='r', encoding='utf8') as fin:
        data_dialogs = fin.readlines()
    print(data_dialogs.count('\n'))
    return


def pair_features():
    with open('../proposed_dataset/test_cross_project/angular.tsv', mode='r', encoding='utf8') as fin, \
        open('../proposed_dataset/test_cross_project/angular_features.tsv', mode='r', encoding='utf8') as ftin:
        utterance_raw = fin.readlines()
        utterance_feat_raw = ftin.readlines()
        fin.close()
        ftin.close()
    with open('../proposed_dataset/test_cross_project/angular_issue.tsv', mode='r', encoding='utf8') as issuein:
        utterance_issue = issuein.readlines()
    issuein.close()
    feat_dialog, feat_add = [], []
    mark_add = False
    # for utterance_each, utterance_feat_each in zip(utterance_raw, utterance_feat_raw):
    #     if utterance_each in utterance_issue and utterance_each != '\n':
    #         feat_add.append(utterance_feat_each)
    #         mark_add = True
    #     elif utterance_each == '\n' and mark_add:
    #         feat_dialog.append(feat_add)
    #         feat_add = []
    #     elif utterance_each not in utterance_issue:
    #         mark_add = False
    count_dialog = 0
    with open('../proposed_dataset/test_cross_project/angular_features_issue.tsv', mode='w', encoding='utf8') as fout:
        for issue_each in utterance_issue:
            if issue_each != '\n':
                index_find = utterance_raw.index(issue_each)
                feat_data = utterance_feat_raw[index_find]
                # feat_add.append(feat_data)
                fout.write(feat_data)
            else:
                count_dialog += 1
                fout.write('\n')
                print(count_dialog)
                # feat_dialog.append(feat_add)
                # feat_add = []
    fout.close()
    return


if __name__ == '__main__':
    read_dialogs()
    # calculate_dialogs()
    # pair_features()