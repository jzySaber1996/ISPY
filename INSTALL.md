# ISPY Installation
This is a installation guideline for "ISPY: Automated Issue Detection and Solution Extraction".

## Dialogue Disentanglement
Before runnning the ISPY prediction, the plain text need to be disentangled into dialogues. 

The code of dialogue disentanglement is available in [disentanglement](./disentanglement). We choose the SOTA model
[irc-disentanglement](https://github.com/jkkummerfeld/irc-disentanglement/zipball/master) to seperate the raw dataset.

The steps of running dialogue disentanglement is as follows:

1. Predict the link of utterances.

2. Extract separate dialogue messages via link graph.

3. Transfer dialogue messages into separate dialogues

## ISPY Model
The 