# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
InferSent models. See https://github.com/facebookresearch/InferSent.
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import logging
import time
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# get models.py from InferSent repo
from models import InferSent

# Set PATHs
PATH_SENTEVAL = '/home/yash/sbert/sentence-transformers/SentEval/'
PATH_TO_DATA = '/home/yash/sbert/sentence-transformers/SentEval/data'
PATH_TO_W2V = '/home/yash/sbert/sentence-transformers/SentEval/src/Other_encoder_models/infersent/GloVe/glove.840B.300d.txt'  # or crawl-300d-2M.vec for V2
MODEL_PATH = '/home/yash/sbert/sentence-transformers/SentEval/src/Other_encoder_models/infersent/encoder/infersent1.pkl'
V = 1 # version of InferSent

assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_W2V), \
    'Set MODEL and GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval


def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)


def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Load InferSent model
    device = torch.device('cuda:1')
    save_path = '/home/yash/sbert/sentence-transformers/SentEval/src/Other_encoder_models/results/'
    results_embs={}
    emb_time={}
    save=True 
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    model = InferSent(params_model).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.set_w2v_path(PATH_TO_W2V)
    emb_name = 'infersent'
    
    params_senteval['infersent'] = model.to(device)

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    start = time.time()
    transfer_tasks = [
                     'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                      ]
    results = se.eval(transfer_tasks)
    end = time.time()
    print ("Results for embedding '{}'".format(emb_name))
    print(results)
    print('Time took on task %s : %.1f. seconds' % (transfer_tasks, end - start))
    results_embs[emb_name] = results
    emb_time[emb_name] = end - start
    print(results)
    if save:
        final_path = save_path+"infersent.json"
        if not os.path.exists(final_path):
                os.mkdir(final_path)
        with open(final_path,"w") as f:
            f.dumps(results_embs)
            f.dumps(emb_time)
    print(results_embs)
    print("done")
