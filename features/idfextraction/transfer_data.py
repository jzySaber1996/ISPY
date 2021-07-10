import xlrd

projects = ['Angular', 'Appium', 'Deeplearning4j', 'Docker', 'Ethereum', 'Nodejs', 'Gitter', 'Typescript']
remark_pos = [7, 7, 8, 7, 7, 7, 7, 7]
# utterances = []
for project, remark in zip(projects, remark_pos):
    project_file = '../../data/issuedialog/' + project + '.tsv'
    workbook = xlrd.open_workbook('../../data/data_new.xls')
    sheet = workbook.sheet_by_name(project)
    data_utterances = sheet.col_values(1, 1, sheet.nrows)
    data_remarks = sheet.col_values(remark, 1, sheet.nrows)
    data_issues = sheet.col_values(2, 1, sheet.nrows)
    with open(project_file, 'w', encoding='utf8') as f_out:
        for index, (data_utterance, data_remark, data_issue) in enumerate(zip(data_utterances, data_remarks, data_issues)):
            if data_remark != '' or data_utterance == '':
                continue
            issue_mark = 0
            if data_issue != '':
                issue_mark = 1
            conversation = data_utterance.split('\n')
            label_list = [each_utterance[:each_utterance.index('=')].replace('+', '')
                            for each_utterance in conversation if '>' in each_utterance]
            mark_list = [each_utterance[0] for each_utterance in conversation if '>' in each_utterance]
            answer_list = []
            for mark in mark_list:
                if mark == '+':
                    answer_list.append(1)
                else:
                    answer_list.append(0)
            text_list = [each_utterance[each_utterance.index('>') + 2:] + ' __eou__'
                            for each_utterance in conversation if '>' in each_utterance]
            user_list = [each_utterance[each_utterance.index('<') + 1:each_utterance.index('>')]
                         for each_utterance in conversation if '>' in each_utterance]
            # utterances += conversation
            user_agent = []
            for user in user_list:
                if user == user_list[0]:
                    user_agent.append('user')
                else:
                    user_agent.append('agent')
            for label, text, ua, answer in zip(label_list, text_list, user_agent, answer_list):
                f_out.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(label, text, ua, str(answer), str(issue_mark)))
                # f_out.write(label + '\t' + text + '\t' + ua + '\t' + str(answer) + '\n')
            f_out.write('\n')
        f_out.close()