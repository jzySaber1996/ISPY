# ISPY Installation
This is a installation guideline for "ISPY: Automated Issue Detection and Solution Extraction".

## Dialogue Disentanglement
Before runnning the ISPY prediction, the plain text need to be disentangled into dialogues. 

The code of dialogue disentanglement is available in [disentanglement](./disentanglement). We choose the SOTA model
[irc-disentanglement](https://github.com/jkkummerfeld/irc-disentanglement/zipball/master) to seperate the raw dataset.

The steps of running dialogue disentanglement is as follows:

### Step 1: Preprocessing.
- Generate the `.ascii.txt` file (Attention: this function may not modify the name of `.txt`.): 
```python
> python3 make-txt.py <filedir>
```
- Replace `.txt` into `.ascii.txt`:
```python
> rename txt ascii.txt <filedir>
```
- Generate `.tok.txt`:
```python
>  python dstc8-tokenise.py --vocab <vocabdir> --output-suffix .tok <filedir> <filedir>
```

### Step 2: Disentanglement. Predict the link of utterances.
```python
> python3 disentangle.py \                          
  <filename>.1\
  --model example-train.dy.model \
  --test <tokdir> \
  --test-start 0 \
  --test-end 5000 \
  --hidden 512 \
  --layers 2 \
  --nonlin softsign \
  --word-vectors <vecdir> \
  > <filename>.out 2><filename>.1.err
```

### Step 3: Extract separate dialogue messages via link graph.
```python
> python3 graph-to-messages.py <filedir> <filedir>
```

### Step 4: Transfer dialogue messages into separate dialogues.
-  If you need to check the disentanglement result, use this command:
```python
> python3 merge_file.py <filedir>
```
- Otherwise, you can skip this Step 4 and start to predict the ISPY models.

## ISPY Model
The ISPY includes two basic prediction models: `issue_classification.py` and `solution_extraction.py`.

Both models are available at [models](./models), and our SOTA models are reserved in [sota_model](./sota_model). We strongly recommend users of ISPY to retrain these two models on new dataset and submit performance issues to us.

### Step 1: Build up ISPY-oriented dataset.
The preprocessing source code is available in [predicted_is_pairs](./predicted_is_pairs). use this command to construct the dataset:
```python
> python3 reformat_dialogs.py <filename>.out
```
The dataset will be constructed as `.tsv` files in [data](./data) directory.

### Step 2: Issue-Solution prediction.
Enter models directory. Execute both issue and solution models in sequence.
```python
> python3 issue_classification.py
> python3 solution_extraction.py
```

The extracted issue-solution pairs will saved in [data/result_data](./data/result_data)

## ISPY Installation in PyCharm
Our model is built up in PyCharm. We welcome you to modify and optimize our model by using PyCharm edition.