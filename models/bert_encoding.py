from keras_bert import get_pretrained, PretrainedList, get_checkpoint_paths

model_path = get_pretrained(PretrainedList.wwm_uncased_large)
paths = get_checkpoint_paths(model_path)
print(paths.config, paths.checkpoint, paths.vocab)
# C:\Users\52993\.keras\datasets\multi_cased_L-12_H-768_A-12

# from keras_bert import extract_embeddings, POOL_NSP, POOL_MAX
#
# model_path = 'C:\\Users\\52993\\.keras\\datasets\\multi_cased_L-12_H-768_A-12'
# # texts = ['all work and no play', 'makes jack a dull boy~']
# texts = [
#     ('all work and no play', 'makes jack a dull boy'),
#     ('makes jack a dull boy', 'all work and no play'),
#     ('makes jack', 'no play'),
#     ('a dull boy', 'all work'),
#
# ]
#
# embeddings = extract_embeddings(model_path, texts, output_layer_num=1, poolings=[POOL_NSP, POOL_MAX])
# print(embeddings)