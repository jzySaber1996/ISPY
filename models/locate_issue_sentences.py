from pprint import pprint
from features.features.cosine_similarity import cosine_similarity
from features.features.content_features import *
from features.features.user_features import *
from features.features.structural_features import *
from features.features.sentiment_features import *
from features.data_helper import *
SYMBOL_FILTER = ['JK', 'GG']
SYMBOL_NOT_USER = ['IG']

def new_classification(list_ins, list_outs):
    for i, ins_addr in enumerate(list_ins):
        with open(ins_addr, encoding='utf8') as fin, open(list_outs[i], mode='w', encoding='utf8') as fout:
            lines = fin.readlines()
            # data_temp = lines[-1]
            result_lines, dialog_lines = [], []
            for line in lines:
                if line != '\n':
                    data_features = line.replace('\n', '').split('\t')
                    if data_features[3] == '1':
                        line_temp = line.replace('\n', '') + '\t' + '1' + '\n'
                        result_lines.append(line_temp)
                    if data_features[5] == '1':
                        line_temp = line.replace('\n', '') + '\t' + '2' + '\n'
                        result_lines.append(line_temp)
                    if data_features[3] == '0' and data_features[5] == '0':
                        line_temp = line.replace('\n', '') + '\t' + '0' + '\n'
                        result_lines.append(line_temp)
                else:
                    result_lines.append('\n')
            for result_line in result_lines:
                fout.write(result_line)
            fout.close()
            fin.close()
    return


def locate_sentences(list_ins, list_outs, feat_files, feat_out_files):
    # # feat_files = ['../data/issuedialog/train_features.tsv', '../data/issuedialog/test_features.tsv', '../data/issuedialog/valid_features.tsv']
    # feat_files = ['../data/issuedialog/train_features_origin.tsv', '../data/issuedialog/test_features_origin.tsv', '../data/issuedialog/valid_features_origin.tsv']
    # # feat_out_files = ['../data/issuedialog/train_feat_new.tsv', '../data/issuedialog/test_feat_new.tsv', '../data/issuedialog/valid_feat_new.tsv']
    # feat_out_files = ['../data/issuedialog/train_feat_new_origin.tsv', '../data/issuedialog/test_feat_new_origin.tsv', '../data/issuedialog/valid_feat_new_origin.tsv']

    pos_file = '../data/positive-words.txt'
    neg_file = '../data/negative-words.txt'
    idf_file = '../data/idf.tsv'
    term_to_idf_dict = init_tf_idf_dict(idf_file)
    pos_dict, neg_dict = load_sentiment_lexicon(pos_file, neg_file)
    for i, (ins_addr, feat_addr) in enumerate(zip(list_ins, feat_files)):
        with open(ins_addr, encoding='utf8') as fin, open(list_outs[i], mode='w', encoding='utf8') as fout, \
                open(feat_addr, encoding='utf8') as feat_in, open(feat_out_files[i], mode='w', encoding='utf8') as featout:
            result_lines, dialog_lines = [], []
            first_line = ''
            mark_former_new, mark_first_sentence = True, False
            mark_new_sentence = False
            mark_issue = '0'
            lines = fin.readlines()
            line_feats = feat_in.readlines()
            start_info, feature_start = '', ''
            for line, line_feat in zip(lines, line_feats):
                if line != '\n':
                    if mark_former_new:
                        mark_first_sentence = True
                    if mark_first_sentence:
                        data_element = line.replace('\n', '').split('\t')
                        if (data_element[2] == 'user' and data_element[2] not in SYMBOL_NOT_USER) or \
                                (data_element[2] == 'agent' and data_element[0] in SYMBOL_FILTER):
                            mark_issue = data_element[4]
                            first_line += data_element[1].replace('__eou__', '') + ' '
                            mark_first_sentence = True
                        else:
                            first_line += '__eou__'
                            mark_first_sentence = False
                            mark_new_sentence = True
                            _, init_sim, thread_sim = cosine_similarity("", first_line, term_to_idf_dict)
                            qm = question_mark(first_line)
                            dup = duplicate(first_line)
                            wh = W5H1(first_line)

                            # structural features
                            abs_pos = [idx + 1 for idx in range(len(first_line))]
                            norm_pos = [pos / len(first_line) for pos in abs_pos]
                            length, unique_length, unique_stemmed_length = post_length(first_line)

                            # user features
                            #                 ua = user_auth(affiliations)
                            is_starter = [1]

                            # sentiment based features
                            thx = thank(first_line)
                            exclam_mark = exclamation_mark(first_line)
                            vf = ve_feedback(first_line)
                            ss = sentiment_scores(first_line)
                            lexicon_counts = lexicon(first_line, pos_dict, neg_dict)

                            # write to file
                            for i, utterance in enumerate([first_line]):
                                #                     try:
                                feature_start = '{}\t{:.4f} {:.4f} {} {} {} {} {:.4f} {} {} {} {} {} {} {} {} {}\n'.format(
                                    'OQ', init_sim[i], thread_sim[i], qm[i], dup[i], ' '.join(wh[i]), abs_pos[i], norm_pos[i],
                                    length[i],
                                    unique_length[i],
                                    unique_stemmed_length[i],
                                    is_starter[i],
                                    thx[i],
                                    exclam_mark[i],
                                    vf[i],
                                    ' '.join(ss[i]),
                                    ' '.join(lexicon_counts[i]),
                                )
                            start_info = '{}\t{}\t{}\t{}\t{}\t{}\n'.format('OQ', first_line, 'user', '0', mark_issue, mark_issue)
                            dialog_lines.append([start_info, feature_start])
                            first_line = ''
                    if not mark_first_sentence:
                        dialog_lines.append([line.replace('\n', '') + '\t0\n', line_feat])
                    # if not mark_first_sentence and not mark_new_sentence:
                    #     temp=1
                    mark_former_new = False
                else:
                    if first_line != '':
                        first_line += '__eou__'
                        mark_first_sentence = False
                        mark_new_sentence = True
                        _, init_sim, thread_sim = cosine_similarity("", first_line, term_to_idf_dict)
                        qm = question_mark(first_line)
                        dup = duplicate(first_line)
                        wh = W5H1(first_line)

                        # structural features
                        abs_pos = [idx + 1 for idx in range(len(first_line))]
                        norm_pos = [pos / len(first_line) for pos in abs_pos]
                        length, unique_length, unique_stemmed_length = post_length(first_line)

                        # user features
                        #                 ua = user_auth(affiliations)
                        is_starter = [1]

                        # sentiment based features
                        thx = thank(first_line)
                        exclam_mark = exclamation_mark(first_line)
                        vf = ve_feedback(first_line)
                        ss = sentiment_scores(first_line)
                        lexicon_counts = lexicon(first_line, pos_dict, neg_dict)

                        # write to file
                        for i, utterance in enumerate([first_line]):
                            #                     try:
                            feature_start = '{}\t{:.4f} {:.4f} {} {} {} {} {:.4f} {} {} {} {} {} {} {} {} {}\n'.format(
                                'OQ', init_sim[i], thread_sim[i], qm[i], dup[i], ' '.join(wh[i]), abs_pos[i],
                                norm_pos[i],
                                length[i],
                                unique_length[i],
                                unique_stemmed_length[i],
                                is_starter[i],
                                thx[i],
                                exclam_mark[i],
                                vf[i],
                                ' '.join(ss[i]),
                                ' '.join(lexicon_counts[i]),
                            )
                        start_info = '{}\t{}\t{}\t{}\t{}\t{}\n'.format('OQ', first_line, 'user', '0', mark_issue,
                                                                       mark_issue)
                        dialog_lines.append([start_info, feature_start])
                        first_line = ''
                    result_lines.append(dialog_lines)
                    dialog_lines = []
                    mark_former_new = True

            uoas = []
            for each_dialogues in result_lines:
                for each_sentence in each_dialogues:
                    fout.write(each_sentence[0])
                    featout.write(each_sentence[1])
                fout.write('\n')
                featout.write('\n')
            fout.close()
            featout.close()
    return


def select_only_issue_dataset(input_files, input_feat_files, output_files, output_feat_files):
    # input_files = ['../data/issuedialog/train_new.tsv', '../data/issuedialog/test_new.tsv',
    #                '../data/issuedialog/valid_new.tsv']
    # input_feat_files = ['../data/issuedialog/train_feat_new.tsv', '../data/issuedialog/test_feat_new.tsv',
    #                     '../data/issuedialog/valid_feat_new.tsv']
    # output_files = ['../data/issuedialog/train_new_solution.tsv', '../data/issuedialog/test_new_solution.tsv',
    #                 '../data/issuedialog/valid_new_solution.tsv']
    # output_feat_files = ['../data/issuedialog/train_feat_new_solution.tsv', '../data/issuedialog/test_feat_new_solution.tsv',
    #                 '../data/issuedialog/valid_feat_new_solution.tsv']
    for input_file, input_feat_file, output_file, output_feat_file in zip(input_files, input_feat_files, output_files, output_feat_files):
        with open(input_file, encoding='utf8') as fin, open(input_feat_file, encoding='utf8') as featin,\
                open(output_file, mode='w', encoding='utf8') as fout, open(output_feat_file, mode='w', encoding='utf8') as featout:
            input_data = fin.readlines()
            input_feat_data = featin.readlines()
            latest_enter = 0
            count_length = 0
            for input_val, input_feat_val in zip(input_data, input_feat_data):
                if input_val == '\n' and latest_enter == 0:
                    fout.write(input_val)
                    featout.write(input_feat_val)
                    latest_enter = 1
                    count_length = 0
                elif input_val != '\n':
                    input_val_list = input_val.replace('\n', '').split('\t')
                    if input_val_list[4] == '0':
                        latest_enter = 1
                    elif input_val_list[4] == '1' and count_length <= 8:
                        fout.write(input_val)
                        featout.write(input_feat_val)
                        latest_enter = 0
                        count_length += 1
            print(count_length)

    return


if __name__ == '__main__':
    projects = ['Angular', 'Appium', 'Deeplearning4j', 'Docker', 'Ethereum', 'Nodejs', 'Gitter', 'Typescript']
    for project in projects:
        # locate_sentences([f'../data/issuedialog_projects/train_origin_{project}.tsv', f'../data/issuedialog_projects/test_origin_{project}.tsv', f'../data/issuedialog_projects/valid_origin_{project}.tsv'],
        #                  [f'../data/issuedialog_projects/train_new_origin_{project}.tsv', f'../data/issuedialog_projects/test_new_origin_{project}.tsv', f'../data/issuedialog_projects/valid_new_origin_{project}.tsv'],
        #                  [f'../data/issuedialog_projects/train_features_origin_{project}.tsv', f'../data/issuedialog_projects/test_features_origin_{project}.tsv', f'../data/issuedialog_projects/valid_features_origin_{project}.tsv'],
        #                  [f'../data/issuedialog_projects/train_feat_new_origin_{project}.tsv', f'../data/issuedialog_projects/test_feat_new_origin_{project}.tsv', f'../data/issuedialog_projects/valid_feat_new_origin_{project}.tsv'])
        # new_classification(['../data/issuedialog/train_new.tsv', '../data/issuedialog/test_new.tsv', '../data/issuedialog/valid_new.tsv'],
        #                  ['../data/issuedialog/train_new_2.tsv', '../data/issuedialog/test_new_2.tsv', '../data/issuedialog/valid_new_2.tsv'])
        select_only_issue_dataset([f'../data/issuedialog_projects_v1/train_new_origin_{project}.tsv', f'../data/issuedialog_projects_v1/test_new_origin_{project}.tsv', f'../data/issuedialog_projects_v1/valid_new_origin_{project}.tsv'],
                                  [f'../data/issuedialog_projects_v1/train_feat_new_origin_{project}.tsv', f'../data/issuedialog_projects_v1/test_feat_new_origin_{project}.tsv', f'../data/issuedialog_projects_v1/valid_feat_new_origin_{project}.tsv'],
                                  [f'../data/issuedialog_projects_vsolution/train_new_solution_{project}.tsv', f'../data/issuedialog_projects_vsolution/test_new_solution_{project}.tsv', f'../data/issuedialog_projects_vsolution/valid_new_solution_{project}.tsv'],
                                  [f'../data/issuedialog_projects_vsolution/train_feat_new_solution_{project}.tsv', f'../data/issuedialog_projects_vsolution/test_feat_new_solution_{project}.tsv', f'../data/issuedialog_projects_vsolution/valid_feat_new_solution_{project}.tsv'])