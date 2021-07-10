import xlrd


def calculate_data(project_name):
    workbook = xlrd.open_workbook('../data/data_new.xls')
    sheet = workbook.sheet_by_name(project_name)
    dialog_messages = sheet.col_values(1, 1, sheet.nrows)
    issue_messages = sheet.col_values(2, 1, sheet.nrows)
    len_non, len_ = 0, 0
    issue_neg = issue_messages.count('')
    issue_pos = len(issue_messages) - issue_neg
    for dialog, issue in zip(dialog_messages, issue_messages):
        len_dialog = dialog.count('\n') + 1
        if issue == '':
            len_non += len_dialog
        else:
            len_ += len_dialog
    if issue_neg > issue_pos:
        len_ret = (issue_neg - issue_pos) * (len_/issue_pos) + len_ + len_non
    else:
        len_ret = (issue_pos - issue_neg) * (len_non/issue_neg) + len_ + len_non
    print('{}: {}, {}, {}'.format(project_name, len_ret, issue_pos, len_))
    # print()
    return

if __name__ == '__main__':
    projects = ['Angular', 'Appium', 'Deeplearning4j', 'Docker', 'Ethereum', 'Nodejs', 'Gitter', 'Typescript']
    for project in projects:
        calculate_data(project)