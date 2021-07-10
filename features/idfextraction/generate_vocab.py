from sklearn.feature_extraction.text import CountVectorizer
from features.data_helper import *
import xlrd

def generate_vocab(data):

    # count_vect = CountVectorizer(stop_words='english')
    count_vect = CountVectorizer(tokenizer=tokenizer, stop_words='english'  )
    counts = count_vect.fit_transform(data)

    return count_vect.vocabulary_

if __name__ == '__main__':
    # conn_title = connect_db()
    # conn_utter = connect_db()
    #
    # sql_title = 'select title from titles_final'
    # sql_utter = 'select utterance from contents_final'
    #
    # with conn_title.cursor() as cursor_title, conn_utter.cursor() as cursor_utter:
    #     cursor_title.execute(sql_title)
    #     titles = [row['title'] for row in cursor_title.fetchall()]
    #
    #     cursor_utter.execute(sql_utter)
    #     utterances = [row['utterance'] for row in cursor_utter.fetchall()]

    vocab_file = '../../data/vocab.tsv'
    projects = ['Angular', 'Appium', 'Deeplearning4j', 'Docker', 'Ethereum', 'Nodejs', 'Gitter', 'Typescript']
    remark_pos = [6, 6, 7, 6, 6, 6, 6, 6]
    utterances = []
    for project, remark in zip(projects, remark_pos):
        workbook = xlrd.open_workbook('../../data/data.xls')
        sheet = workbook.sheet_by_name(project)
        data_utterances = sheet.col_values(1, 1, sheet.nrows)
        data_remarks = sheet.col_values(remark, 1, sheet.nrows)
        for data_utterance, data_remark in zip(data_utterances, data_remarks):
            if data_remark != '' or data_utterance == '':
                continue
            conversation = data_utterance.split('\n')
            conversation = [each_utterance[each_utterance.index('>') + 1:] for each_utterance in conversation if '>' in each_utterance]
            utterances += conversation
    vocab = generate_vocab(map(clean_str, utterances))#titles + utterances

    with open(vocab_file, 'w') as vocab_output:
        for term in vocab:
            vocab_output.write('{0}\t{1}\n'.format(term, vocab[term]))