#!/usr/bin/env python3

import argparse
import random
import sys
import string
import time
import glob

import numpy as np

FEATURES = 77
N_FOLD = 10

parser = argparse.ArgumentParser(description='IRC Conversation Disentangler.')

# General arguments
parser.add_argument('--prefix', default="C:/Users/52993/Desktop/ESEJ/IJCAI+ACL/disentanglement/disentanglement/example-train", help="Start of names for files produced.")

# Data arguments
parser.add_argument('--train', nargs="+", help="Training files, e.g. train/*annotation.txt")# , default='../data/retrain/train/*annotation.txt'
parser.add_argument('--dev', nargs="+", help="Development files, e.g. dev/*annotation.txt")# , default='../data/retrain/dev/*annotation.txt'
parser.add_argument('--test', nargs="+", help="Test files, e.g. test/*annotation.txt", default='../data/retrain/test/test_Typescript/*ascii*')# , default='../data/test_disentangled/*ascii*'
parser.add_argument('--test-start', type=int, help="The line to start making predictions from in each test file.", default=0)#, default=0
parser.add_argument('--test-end', type=int, help="The line to stop making predictions on in each test files.", default=1280)# , default=105930
parser.add_argument('--model', help="A file containing a trained model", default='C:\\Users\\52993\\Desktop\\ESEJ\\IJCAI+ACL\\disentanglement\\disentanglement\\example-train.dy_Typescript.model')#
parser.add_argument('--random-sample', help="Train on only a random sample of the data with this many examples.")

# Model arguments
parser.add_argument('--hidden', default=512, type=int, help="Number of dimensions in hidden vectors.")
parser.add_argument('--word-vectors', default='../data/glove-ubuntu.txt', help="File containing word embeddings.")
parser.add_argument('--layers', default=2, type=int, help="Number of hidden layers in the model")
parser.add_argument('--nonlin', choices=["tanh", "cube", "logistic", "relu", "elu", "selu", "softsign", "swish", "linear"], default='softsign', help="Non-linearity type.")

# Inference arguments
parser.add_argument('--max-dist', default=101, type=int, help="Maximum number of messages to consider when forming a link (count includes the current message).")
parser.add_argument('--dynet-autobatch', action='store_true', help="Use dynet autobatching.")

# Training arguments
parser.add_argument('--report-freq', default=1000, type=int, help="How frequently to evaluate on the development set.")
parser.add_argument('--epochs', default=15, type=int, help="Maximum number of epochs.")
parser.add_argument('--opt', choices=['sgd', 'mom'], default='sgd', help="Optimisation method.")
parser.add_argument('--seed', default=10, type=int, help="Random seed.")
parser.add_argument('--weight-decay', default=1e-7, type=float, help="Apply weight decay.")
parser.add_argument('--learning-rate', default=0.018804, type=float, help="The initial learning rate.")
parser.add_argument('--learning-decay-rate', default=0.103, type=float, help="The rate at which the learning rate decays.")
parser.add_argument('--momentum', default=0.1, type=float, help="Hyperparameter for momentum.")
parser.add_argument('--drop', default=0.0, type=float, help="Dropout, applied to inputs only.")
parser.add_argument('--clip', default=3.740, type=float, help="Gradient clipping.")

args = parser.parse_args()

WEIGHT_DECAY = args.weight_decay
HIDDEN = args.hidden
LEARNING_RATE = args.learning_rate
LEARNING_DECAY_RATE = args.learning_decay_rate
MOMENTUM = args.momentum
EPOCHS = args.epochs
DROP = args.drop
MAX_DIST = args.max_dist

def header(args, out=sys.stdout):
    head_text = "# "+ time.ctime(time.time())
    head_text += "\n# "+ ' '.join(args)
    for outfile in out:
        print(head_text, file=outfile)

log_file = open(args.prefix +".log", 'w')
header(sys.argv, [log_file, sys.stdout])

import dynet_config
batching = 1 if args.dynet_autobatch else 0
dynet_config.set(mem=512, autobatch=batching, weight_decay=WEIGHT_DECAY, random_seed=args.seed)
dynet_config.set_gpu(True)
import dynet as dy 

from src.reserved_words import reserved


###############################################################################

def update_user(users, user):
    if user in reserved:
        return
    all_digit = True
    for char in user:
        if char not in string.digits:
            all_digit = False
    if all_digit:
        return
    users.add(user.lower())

def update_users(line, users):
    if len(line) < 2:
        return
    user = line[1]
    if user in ["Topic", "Signoff", "Signon", "Total", "#ubuntu"
            "Window", "Server:", "Screen:", "Geometry", "CO,",
            "Current", "Query", "Prompt:", "Second", "Split",
            "Logging", "Logfile", "Notification", "Hold", "Window",
            "Lastlog", "Notify", 'netjoined:']:
        # Ignore as these are channel commands
        pass
    else:
        if line[0].endswith("==="):
            parts = ' '.join(line).split("is now known as")
            if len(parts) == 2 and line[-1] == parts[-1].strip():
                user = line[-1]
        elif line[0][-1] == ']':
            if user[0] == '<':
                user = user[1:]
            if user[-1] == '>':
                user = user[:-1]

        user = user.lower()
        update_user(users, user)
        # This is for cases like a user named |blah| who is
        # refered to as simply blah
        core = [char for char in user]
        while len(core) > 0 and core[0] in string.punctuation:
            core.pop(0)
        while len(core) > 0 and core[-1] in string.punctuation:
            core.pop()
        core = ''.join(core)
        update_user(users, core)

# Names two letters or less that occur more than 500 times in the data
common_short_names = {"ng", "_2", "x_", "rq", "\\9", "ww", "nn", "bc", "te",
"io", "v7", "dm", "m0", "d1", "mr", "x3", "nm", "nu", "jc", "wy", "pa", "mn",
"a_", "xz", "qr", "s1", "jo", "sw", "em", "jn", "cj", "j_"}

def get_targets(line, users):
    targets = set()
    for token in line[2:]:
        token = token.lower()
        user = None
        if token in users and len(token) > 2:
            user = token
        else:
            core = [char for char in token]
            while len(core) > 0 and core[-1] in string.punctuation:
                core.pop()
                nword = ''.join(core)
                if nword in users and (len(core) > 2 or nword in common_short_names):
                    user = nword
                    break
            if user is None:
                while len(core) > 0 and core[0] in string.punctuation:
                    core.pop(0)
                    nword = ''.join(core)
                    if nword in users and (len(core) > 2 or nword in common_short_names):
                        user = nword
                        break
        if user is not None:
            targets.add(user)
    return targets

def lines_to_info(text_ascii):
    users = set()
    for line in text_ascii:
        update_users(line, users)

    chour = 12
    cmin = 0
    info = []
    target_info = {}
    nexts = {}
    count_line = 0
    for line_no, line in enumerate(text_ascii):
        # if count_line == 15354:
        #     print('temp')
        # if len(line) > 0 and line[0].startswith("[") and line[0][1].isdigit() and int(line[0][1])==2:
        if len(line) > 0 and line[0].startswith("[") and line[0][1].isdigit():
            user = line[2][1:-1]
            nexts.setdefault(user, []).append(line_no)
        # count_line += 1
        # print(count_line)

    prev = {}
    for line_no, line in enumerate(text_ascii):
        if count_line == 24694:
            print("temp")
        if len(line) > 2:
            user = line[2]
        else:
            user = ''
        count_line += 1
        print(count_line)
        system = True
        if len(line) > 0 and line[0].startswith("[") and line[0][1].isdigit() and int(line[0][1]) == 2:
            # chour = int(line[0][1:3])
            # cmin = int(line[0][4:6])
            # user = user[1:-1]
            # print(line[0][12:14])
            # chour = int(line[0][12:14])
            # cmin = int(line[0][15:17])
            # print(line[1][0:2])
            chour = int(line[1][0:2])
            cmin = int(line[1][3:5])
            user = user[1:-1]
            system = False
        is_bot = (user == 'ubottu' or user == 'ubotu')
        targets = get_targets(line, users)
        for target in targets:
            target_info.setdefault((user, target), []).append(line_no)
        last_from_user = prev.get(user, None)
        if not system:
            prev[user] = line_no
        next_from_user = None
        if user in nexts:
            while len(nexts[user]) > 0 and nexts[user][0] <= line_no:
                nexts[user].pop(0)
            if len(nexts[user]) > 0:
                next_from_user = nexts[user][0]

        info.append((user, targets, chour, cmin, system, is_bot, last_from_user, line, next_from_user))

    return info, target_info

def get_time_diff(info, a, b):
    if a is None or b is None:
        return -1
    if a > b:
        t = a
        a = b
        b = t
    ahour = info[a][2]
    amin = info[a][3]
    bhour = info[b][2]
    bmin = info[b][3]
    if ahour == bhour:
        return bmin - amin
    if bhour < ahour:
        bhour += 24
    return (60 - amin) + bmin + 60*(bhour - ahour - 1)

cache = {}
def get_features(name, query_no, link_no, text_ascii, text_tok, info, target_info, do_cache):
    global cache
    if (name, query_no, link_no) in cache:
        return cache[name, query_no, link_no]

    features = []

    quser, qtargets, qhour, qmin, qsystem, qis_bot, qlast_from_user, qline, qnext_from_user = info[query_no]
    luser, ltargets, lhour, lmin, lsystem, lis_bot, llast_from_user, lline, lnext_from_user = info[link_no]

    # General information about this sample of data
    # Year
    for i in range(2004, 2018):
        features.append(str(i) in name)
    # Number of messages per minute
    start = None
    end = None
    for i in range(len(text_ascii)):
        if start is None and text_ascii[i][0].startswith("["):
            start = i
        if end is None and i > 0 and text_ascii[-i][0].startswith("["):
            end = len(text_ascii) - i - 1
        if start is not None and end is not None:
            break
    diff = get_time_diff(info, start, end)
    msg_per_min = len(text_ascii) / max(1, diff)
    cutoffs = [-1, 1, 3, 10, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= msg_per_min < end)

    # Query
    #  - Normal message or system message
    features.append(qsystem)
    #  - Hour of day
    features.append(qhour / 24)
    #  - Is it targeted
    features.append(len(qtargets) > 0)
    #  - Is there a previous message from this user?
    features.append(qlast_from_user is not None)
    #  - Did the previous message from this user have a target?
    if qlast_from_user is None:
        features.append(False)
    else:
        features.append(len(info[qlast_from_user][1]) > 0)
    #  - How long ago was the previous message from this user in messages?
    dist = -1 if qlast_from_user is None else query_no - qlast_from_user
    cutoffs = [-1, 0, 1, 5, 20, 1000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= dist < end)
    #  - How long ago was the previous message from this user in minutes?
    time = get_time_diff(info, query_no, qlast_from_user)
    cutoffs = [-1, 0, 2, 10, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= time < end)
    #  - Are they a bot?
    features.append(qis_bot)

    # Link
    #  - Normal message or system message
    features.append(lsystem)
    #  - Hour of day
    features.append(lhour / 24)
    #  - Is it targeted
    features.append(link_no != query_no and len(ltargets) > 0)
    #  - Is there a previous message from this user?
    features.append(link_no != query_no and llast_from_user is not None)
    #  - Did the previous message from this user have a target?
    if link_no == query_no or llast_from_user is None:
        features.append(False)
    else:
        features.append(len(info[llast_from_user][1]) > 0)
    #  - How long ago was the previous message from this user in messages?
    dist = -1 if llast_from_user is None else link_no - llast_from_user
    cutoffs = [-1, 0, 1, 5, 20, 1000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(link_no != query_no and start <= dist < end)
    #  - How long ago was the previous message from this user in minutes?
    time = get_time_diff(info, link_no, llast_from_user)
    cutoffs = [-1, 0, 2, 10, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= time < end)
    #  - Are they a bot?
    features.append(lis_bot)
    #  - Is the message after from the same user?
    features.append(link_no != query_no and link_no + 1 < len(info) and luser == info[link_no + 1][0])
    #  - Is the message before from the same user?
    features.append(link_no != query_no and link_no - 1 > 0 and luser == info[link_no - 1][0])

    # Both
    #  - Is this a self-link?
    features.append(link_no == query_no)
    #  - How far apart in messages are the two?
    dist = query_no - link_no
    features.append(min(100, dist) / 100)
    features.append(dist > 1)
    #  - How far apart in time are the two?
    time = get_time_diff(info, link_no, query_no)
    features.append(min(100, time) / 100)
    cutoffs = [-1, 0, 1, 5, 60, 10000]
    for start, end in zip(cutoffs, cutoffs[1:]):
        features.append(start <= time < end)
    #  - Does the link target the query user?
    features.append(quser.lower() in ltargets)
    #  - Does the query target the link user?
    features.append(luser.lower() in qtargets)
    #  - none in between from src?
    features.append(link_no != query_no and (qlast_from_user is None or qlast_from_user < link_no))
    #  - none in between from target?
    features.append(link_no != query_no and (lnext_from_user is None or lnext_from_user > query_no))
    #  - previously src addressed target?
    #  - future src addressed target?
    #  - src addressed target in between?
    if link_no != query_no and (quser, luser) in target_info:
        features.append(min(target_info[quser, luser]) < link_no)
        features.append(max(target_info[quser, luser]) > query_no)
        between = False
        for num in target_info[quser, luser]:
            if query_no > num > link_no:
                between = True
        features.append(between)
    else:
        features.append(False)
        features.append(False)
        features.append(False)
    #  - previously target addressed src?
    #  - future target addressed src?
    #  - target addressed src in between?
    if link_no != query_no and (luser, quser) in target_info:
        features.append(min(target_info[luser, quser]) < link_no)
        features.append(max(target_info[luser, quser]) > query_no)
        between = False
        for num in target_info[luser, quser]:
            if query_no > num > link_no:
                between = True
        features.append(between)
    else:
        features.append(False)
        features.append(False)
        features.append(False)
    #  - are they the same speaker?
    features.append(luser == quser)
    #  - do they have the same target?
    features.append(link_no != query_no and len(ltargets.intersection(qtargets)) > 0)
    #  - Do they have words in common?
    ltokens = set(text_ascii[link_no])
    qtokens = set(text_ascii[query_no])
    common = len(ltokens.intersection(qtokens))
    if link_no != query_no and len(ltokens) > 0 and len(qtokens) > 0:
        features.append(common / len(ltokens))
        features.append(common / len(qtokens))
    else:
        features.append(False)
        features.append(False)
    features.append(link_no != query_no and common == 0)
    features.append(link_no != query_no and common == 1)
    features.append(link_no != query_no and common > 1)
    features.append(link_no != query_no and common > 5)
    
    # Convert to 0/1
    final_features = []
    for feature in features:
        if feature == True:
            final_features.append(1.0)
        elif feature == False:
            final_features.append(0.0)
        else:
            final_features.append(feature)

    if do_cache:
        cache[name, query_no, link_no] = final_features
    return final_features


def read_data(filenames, is_test=False):
    instances = []
    done = set()
    for filename in filenames:
        name = filename
        for ending in [".annotation.txt", ".ascii.txt", ".raw.txt", ".tok.txt"]:
            if filename.endswith(ending):
                name = filename[:-len(ending)]
        if name in done:
            continue
        done.add(name)
        text_ascii = [l.strip().split() for l in open(name +".ascii.txt", encoding='utf8')]
        text_tok = []
        for l in open(name +".tok.txt"):
            l = l.strip().split()
            if len(l) > 0 and l[-1] == "</s>":
                l = l[:-1]
            if len(l) == 0 or l[0] != '<s>':
                l.insert(0, "<s>")
            text_tok.append(l)
        info, target_info = lines_to_info(text_ascii)

        links = {}
        if is_test:
            for i in range(args.test_start, min(args.test_end, len(text_ascii))):
                links[i] = []
        else:
            for line in open(name +".annotation.txt"):
                nums = [int(v) for v in line.strip().split() if v != '-']
                links.setdefault(max(nums), []).append(min(nums))
        for link, nums in links.items():
            instances.append((name +".annotation.txt", link, nums, text_ascii, text_tok, info, target_info))
    return instances

def simplify_token(token):
    chars = []
    for char in token:
        #### Reduce sparsity by replacing all digits with 0.
        if char.isdigit():
            chars.append("0")
        else:
            chars.append(char)
    return ''.join(chars)


class DyNetModel():
    def __init__(self):
        super().__init__()

        self.model = dy.ParameterCollection()

        input_size = FEATURES

        # Create word embeddings and initialise
        self.id_to_token = []
        self.token_to_id = {}
        pretrained = []
        if args.word_vectors:
            for line in open(args.word_vectors):
                parts = line.strip().split()
                word = parts[0].lower()
                vector = [float(v) for v in parts[1:]]
                self.token_to_id[word] = len(self.id_to_token)
                self.id_to_token.append(word)
                pretrained.append(vector)
            NWORDS = len(self.id_to_token)
            DIM_WORDS = len(pretrained[0])
            self.pEmbedding = self.model.add_lookup_parameters((NWORDS, DIM_WORDS))
            self.pEmbedding.init_from_array(np.array(pretrained))
            input_size += 4 * DIM_WORDS

        self.hidden = []
        self.bias = []
        self.hidden.append(self.model.add_parameters((HIDDEN, input_size)))
        self.bias.append(self.model.add_parameters((HIDDEN,)))
        for i in range(args.layers - 1):
            self.hidden.append(self.model.add_parameters((HIDDEN, HIDDEN)))
            self.bias.append(self.model.add_parameters((HIDDEN,)))
        self.final_sum = self.model.add_parameters((HIDDEN, 1))

    def __call__(self, query, options, gold, lengths, query_no):
        if len(options) == 1:
            return None, 0

        final = []
        if args.word_vectors:
            qvecs = [dy.lookup(self.pEmbedding, w) for w in query]
            qvec_max = dy.emax(qvecs)
            qvec_mean = dy.average(qvecs)
        for otext, features in options:
            inputs = dy.inputTensor(features)
            if args.word_vectors:
                ovecs = [dy.lookup(self.pEmbedding, w) for w in otext]
                ovec_max = dy.emax(ovecs)
                ovec_mean = dy.average(ovecs)
                inputs = dy.concatenate([inputs, qvec_max, qvec_mean, ovec_max, ovec_mean])
            if args.drop > 0:
                inputs = dy.dropout(inputs, args.drop)
            h = inputs
            for pH, pB in zip(self.hidden, self.bias):
                h = dy.affine_transform([pB, pH, h])
                if args.nonlin == "linear":
                    pass
                elif args.nonlin == "tanh":
                    h = dy.tanh(h)
                elif args.nonlin == "cube":
                    h = dy.cube(h)
                elif args.nonlin == "logistic":
                    h = dy.logistic(h)
                elif args.nonlin == "relu":
                    h = dy.rectify(h)
                elif args.nonlin == "elu":
                    h = dy.elu(h)
                elif args.nonlin == "selu":
                    h = dy.selu(h)
                elif args.nonlin == "softsign":
                    h = dy.softsign(h)
                elif args.nonlin == "swish":
                    h = dy.cmult(h, dy.logistic(h))
            final.append(dy.sum_dim(h, [0]))

        final = dy.concatenate(final)
        nll = -dy.log_softmax(final)
        dense_gold = []
        for i in range(len(options)):
            dense_gold.append(1.0 / len(gold) if i in gold else 0.0)
        answer = dy.inputTensor(dense_gold)
        loss = dy.transpose(answer) * nll
        predicted_link = np.argmax(final.npvalue())

        return loss, predicted_link

    def get_ids(self, words):
        ans = []
        backup = self.token_to_id.get('<unka>', 0)
        for word in words:
            ans.append(self.token_to_id.get(word, backup))
        return ans

def do_instance(instance, train, model, optimizer, do_cache=True):
    name, query, gold, text_ascii, text_tok, info, target_info = instance

    # Skip cases if we can't represent them
    gold = [v for v in gold if v > query - MAX_DIST]
    if len(gold) == 0 and train:
        return 0, False, query

    # Get features
    options = []
    query_ascii = text_ascii[query]
    query_tok = model.get_ids(text_tok[query])
    for i in range(query, max(-1, query - MAX_DIST), -1):
        option_ascii = text_ascii[i]
        option_tok = model.get_ids(text_tok[i])
        features = get_features(name, query, i, text_ascii, text_tok, info, target_info, do_cache)
        options.append((option_tok, features))
    gold = [query - v for v in gold]
    lengths = [len(sent) for sent in options]

    # Run computation
    example_loss, output = model(query_tok, options, gold, lengths, query)
    loss = 0.0
    if train and example_loss is not None:
        example_loss.backward()
        optimizer.update()
        loss = example_loss.scalar_value()
    predicted = output
    matched = (predicted in gold)

    return loss, matched, predicted

###############################################################################

train = []
if args.train:
    filenames = glob.glob(args.train)
    train = read_data(filenames)
dev = []
if args.dev:
    filenames = glob.glob(args.dev)
    dev = read_data(filenames)
test = dev
if args.test:
    filenames = glob.glob(args.test)
    # filenames = args.test
    test = read_data(filenames, True)
if args.random_sample and args.train:
    random.seed(args.seed)
    random.shuffle(train)
    train = train[:int(args.random_sample)]


prev_best = None
if args.train:
    # 'angular',
    project_name_list = ['angular']
    # , 'dl4j', 'ethereum', 'Gitter', 'Typescript', 'nodejs'
    for project_name in project_name_list:
        # Model and optimizer creation
        model = None
        optimizer = None
        scheduler = None
        model = DyNetModel()
        optimizer = None
        print("Model Initialized")
        if args.opt == 'sgd':
            print("SGD Training")
            optimizer = dy.SimpleSGDTrainer(model.model, learning_rate=LEARNING_RATE)
        elif args.opt == 'mom':
            optimizer = dy.MomentumSGDTrainer(model.model, learning_rate=LEARNING_RATE, mom=MOMENTUM)
        print("Opt Initialized")
        optimizer.set_clip_threshold(args.clip)
        train_dev_list = []
        for train_data in train:
            project_from = train_data[0].split('\\')[1].split('.')[0]
            if project_from != project_name:
                train_dev_list.append(train_data)
        step = 0
        length_train_dev = len(train_dev_list)
        length_fold = int(length_train_dev / N_FOLD)
        for i in range(N_FOLD):
            print('-------Fold {}---------'.format(i))
            train_list, dev_list = [], []
            for j in range(N_FOLD):
                if j == i:
                    dev_list += train_dev_list[j * length_fold:(j + 1) * length_fold - 1]
                else:
                    train_list += train_dev_list[j * length_fold:(j + 1) * length_fold - 1]
            for epoch in range(EPOCHS):
                random.shuffle(train_list)

                # Update learning rate
                optimizer.learning_rate = LEARNING_RATE / (1 + LEARNING_DECAY_RATE * epoch)

                # Loop over batches
                loss = 0
                match = 0
                total = 0
                loss_steps = 0
                for instance in train_list:
                    step += 1
                    print('Test Project: {}, Epoch: {}, Step: {}'.format(project_name, epoch, step))
                    dy.renew_cg()
                    ex_loss, matched, _ = do_instance(instance, True, model, optimizer)
                    loss += ex_loss
                    loss_steps += 1
                    if matched:
                        match += 1
                    total += len(instance[2])

                    # Partial results
                    if step % args.report_freq == 0:
                        # Dev pass
                        dev_match = 0
                        dev_total = 0
                        for dinstance in dev_list:
                            dy.renew_cg()
                            _, matched, _ = do_instance(dinstance, False, model, optimizer)
                            if matched:
                                dev_match += 1
                            dev_total += len(dinstance[2])

                        tacc = match / total
                        dacc = dev_match / dev_total
                        print("{} tl {:.3f} ta {:.3f} da {:.3f} from {} {}".format(epoch, loss / loss_steps, tacc, dacc,
                                                                                   dev_match, dev_total))
                        print("{} tl {:.3f} ta {:.3f} da {:.3f} from {} {}".format(epoch, loss / loss_steps, tacc, dacc,
                                                                                   dev_match, dev_total), file=log_file)

                        log_file.flush()

                        if prev_best is None or prev_best[0] < dacc:
                            prev_best = (dacc, epoch)
                            model.model.save(args.prefix + f".dy_{project_name}.model") # Former version: ".dy.model"\
                        if loss / loss_steps <= 0.05:
                            print('Loss <= 0.05')
                            sys.exit(0)

                if prev_best is not None and epoch - prev_best[1] > 5:
                    break

# Load model
model = None
model = DyNetModel()
optimizer = None
print("Model Initialized")
if args.opt == 'sgd':
    print("SGD Training")
    optimizer = dy.SimpleSGDTrainer(model.model, learning_rate=LEARNING_RATE)
elif args.opt == 'mom':
    optimizer = dy.MomentumSGDTrainer(model.model, learning_rate=LEARNING_RATE, mom=MOMENTUM)
print("Opt Initialized")

if prev_best is not None or args.model:
    location = args.model
    if location is None:
        location = args.prefix +".angular.model" # Former version: ".dy.model"
    model.model.populate(location)

# Run on test instances
for instance in test:
    dy.renew_cg()
    _, _, prediction = do_instance(instance, False, model, optimizer, False)
    print("{}:{} {} -".format(instance[0], instance[1], instance[1] - prediction))
    with open(instance[0], 'a') as f:
        f.write("{} {} -".format(instance[1], instance[1] - prediction))
        f.write('\n')

log_file.close()

