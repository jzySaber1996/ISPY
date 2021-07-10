# %%

# extract features from train/valid/test files

# %%

# import nltk
# nltk.download('vader_lexicon')
from pprint import pprint
from features.features.cosine_similarity import cosine_similarity
from features.features.content_features import *
from features.features.user_features import *
from features.features.structural_features import *
from features.features.sentiment_features import *
from features.data_helper import *

# %%

# train_file = '../data/msdialog/train.tsv'
# valid_file = '../data/msdialog/valid.tsv'
# test_file = '../data/msdialog/test.tsv'

# %%

idf_file = '../data/idf.tsv'

# train_feat_file = '../data/msdialog/train_features.tsv'
# valid_feat_file = '../data/msdialog/valid_features.tsv'
# test_feat_file = '../data/msdialog/test_features.tsv'

pos_file = '../data/positive-words.txt'
neg_file = '../data/negative-words.txt'
term_to_idf_dict = init_tf_idf_dict(idf_file)
pos_dict, neg_dict = load_sentiment_lexicon(pos_file, neg_file)

# %%
# projects = ['angular', 'appium', 'deeplearning4j', 'docker', 'ethereum', 'gitter', 'nodejs', 'typescript']
projects = ['materialize', 'springboot', 'webpack']
in_file_list = [f'../proposed_dataset/new_test_cross_project/{project}.tsv' for project in projects]
out_file_list = [f'../proposed_dataset/new_test_cross_project/{project}_features.tsv' for project in projects]
for in_file, out_file in zip(in_file_list, out_file_list):
    with open(in_file, encoding='utf8') as fin, open(out_file, 'w', encoding='utf8') as fout:
        utterances = []
        labels = []
        uoas = []
        for line in fin:
            if line != '\n':
                tokens = line.strip().split('\t')
                labels.append(tokens[0])
                utterances.append(tokens[1])
                uoas.append(tokens[2])
            else:
                # extract features

                # content based features
                _, init_sim, thread_sim = cosine_similarity("", utterances, term_to_idf_dict)
                qm = question_mark(utterances)
                dup = duplicate(utterances)
                wh = W5H1(utterances)

                # structural features
                abs_pos = [idx + 1 for idx in range(len(utterances))]
                norm_pos = [pos / len(utterances) for pos in abs_pos]
                length, unique_length, unique_stemmed_length = post_length(utterances)

                # user features
                #                 ua = user_auth(affiliations)
                is_starter = [1 if uoa == 'user' else 0 for uoa in uoas]

                # sentiment based features
                thx = thank(utterances)
                exclam_mark = exclamation_mark(utterances)
                vf = ve_feedback(utterances)
                ss = sentiment_scores(utterances)
                lexicon_counts = lexicon(utterances, pos_dict, neg_dict)

                # write to file
                for i, utterance in enumerate(utterances):
                    #                     try:
                    fout.write('{}\t{:.4f} {:.4f} {} {} {} {} {:.4f} {} {} {} {} {} {} {} {} {}\n'.format(
                        labels[i],
                        init_sim[i],
                        thread_sim[i],
                        qm[i],
                        dup[i],
                        ' '.join(wh[i]),
                        abs_pos[i],
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
                    ))
                #                     except:
                #                         print(utterance)

                fout.write('\n')
                utterances = []
                labels = []
                uoas = []
