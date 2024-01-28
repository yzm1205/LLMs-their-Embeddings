import os
from pyexpat import model
import sys
import tensorflow as tf

# tf.config.optimizer.set_jit(False)
import pandas as pd
import tensorflow_hub as hub
import numpy as np
import json

# from src.utils import *
import warnings

sys.path.insert(0, "/home/yash/Probing-Sentence-Encoder/models/Infersent/")
from models import InferSent
import torch
from laserembeddings import Laser
from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile, common_texts
import logging
import urllib.request
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

nltk.download("punkt")

# device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# import argparser # add arguments for commandline later.
with open("src/config.json") as config_json:
    config = json.load(config_json)


def embed(sentence, model, encoder_name=""):
    if encoder_name == "USE":
        return model(sentence)
    if encoder_name == "sbert":
        # return [model.encode(i) for i in sentence]
        return model.encode(sentence)
    if encoder_name == "laser":
        return model.embeddings(sentence)
    if encoder_name == "inferSent":
        return model.encode(sentence, tokenize=True)


def USE():
    logging.info("Loading USE Model")
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    use_model = hub.load(module_url)
    print("USE model loaded successfully....")
    return use_model


def SentBert(task=""):
    logging.info("Loading Sentence-Bert(SBERT) model")

    if task == "synonym" or task == "antonym" or task == "jumbling":
        model_sbert = SentenceTransformer(config["model_task"]["SBert"]["other"])
    else:
        model_sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model_sbert


# if pip laser didn't work out then use this.
class LASERMODEL:
    def __init__(self, output_dir):
        logging.info("Loading LASER models ")
        self.download_models("models/Laser")
        DEFAULT_BPE_CODES_FILE = os.path.join(output_dir, "93langs.fcodes")
        DEFAULT_BPE_VOCAB_FILE = os.path.join(output_dir, "93langs.fvocab")
        DEFAULT_ENCODER_FILE = os.path.join(output_dir, "bilstm.93langs.2018-12-26.pt")
        self.embedding_model = Laser(
            DEFAULT_BPE_CODES_FILE, DEFAULT_BPE_VOCAB_FILE, DEFAULT_ENCODER_FILE
        )

    def embeddings(self, x):
        return self.embedding_model.embed_sentences(x, lang="en")

    def download_file(self, url, dest):
        sys.stdout.flush()
        urllib.request.urlretrieve(url, dest)

    def download_models(self, output_dir):
        logger.info("Downloading models into {}".format(output_dir))

        self.download_file(
            "https://dl.fbaipublicfiles.com/laser/models/93langs.fcodes",
            os.path.join(output_dir, "93langs.fcodes"),
        )
        self.download_file(
            "https://dl.fbaipublicfiles.com/laser/models/93langs.fvocab",
            os.path.join(output_dir, "93langs.fvocab"),
        )
        self.download_file(
            "https://dl.fbaipublicfiles.com/laser/models/bilstm.93langs.2018-12-26.pt",
            os.path.join(output_dir, "bilstm.93langs.2018-12-26.pt"),
        )


def infersent(batch):
    logging.info("Loading InferSent Model")
    params_model = {
        "bsize": batch,
        "word_emb_dim": 300,
        "enc_lstm_dim": 2048,
        "pool_type": "max",
        "dpout_model": 0.0,
        "version": 1,
    }
    model_path = (
        "./models/Infersent/encoder/infersent1.pkl"
    )
    model = InferSent(params_model)
    model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        model = model.cuda()
    glove_embedding = (
        "./models/Infersent/GloVe/glove.840B.300d.txt"
    )
    model.set_w2v_path(glove_embedding)
    model.build_vocab_k_words(100000)
    return model


def doc2vec():
    model_path = get_tmpfile(
        "./models/doc2vec/download"
    )

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
    model.save(model_path)
    model = Doc2Vec.load(model_path)
    return model


def main(model_name: str, batch: int = 64, device: int = 0, save: bool = False):
    device_no = torch.device(f'cuda:{device}') if torch.cuda.is_available() else torch.device('cpu')
    if isinstance(batch, str):
        batch = int(batch)
    df = "" #path to dataset
    _len = len(df.sentence)
 
    print("\n*************************************************************")
    print(f"************* Model {model_name} running on device  **************")
    print("*************************************************************\n")

    if model_name == "sbert":
        model = SentBert()

    elif model_name == "USE":
        model = USE()
    elif model_name == "infersent":
        model = infersent(batch)
    elif model_name == "laser":
        model = Laser()
    elif model_name == "d2v":
        model = doc2vec()

    for idx in tqdm(range(0, _len, batch)):
        sentence_batch = df.sentence.iloc[idx : idx + batch]
        affirm_sentence_batch = df.affirm_sentence.iloc[idx : idx + batch]
        sent_batch = ["".join(s) for s in sentence_batch]
        affirm_sent_batch = ["".join(s) for s in affirm_sentence_batch]
        if model_name == "infersent":
            emb1 = model.encode(sent_batch)
            emb2 = model.encode(affirm_sent_batch)
        elif model_name == "laser":
            emb1 = model.embed_sentences(sent_batch, lang="en")
            emb2 = model.embed_sentences(affirm_sent_batch, lang="en")
        elif model_name == "d2v":
            emb1 = [model.infer_vector([sent]) for sent in sent_batch]
            emb2 = [model.infer_vector([sent]) for sent in affirm_sent_batch]

        similarity = cosine_similarity(emb1, emb2)
        df.at[idx : idx + batch - 1, f"{model_name}_sim"] = np.diag(similarity)

    if save:
        df.to_csv(os.path.join(SAVE_PATH, f"blanco_{model_name}_llm.csv"))


if __name__ == "__main__":
  
    main("infersent", batch=32, save=False)
    print("done!!!")
