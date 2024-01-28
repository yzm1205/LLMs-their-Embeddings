from email.policy import default
import torch
from transformers import BloomModel,AutoTokenizer,GPTNeoModel
import json
from typing import Tuple, Union, List
import sys
import openai
from models.llama2 import llama2
from utils import full_path

with open("./key.json","r",encoding="utf-8") as f:
    keys = json.load(f)

hf_auth = keys["hf_llama_2"]


class ModelLoader:
    def __init__(self, model_name:str , default_gpu="cuda") -> None:
        self.device =default_gpu
        self.is_valid_model = True
        
        if model_name in ["Bloom","bloom"]:
            # Load Bloom model and tokenizer during the instantiation of the class
            self.model = BloomModel.from_pretrained("bigscience/bloom-560m").to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
        
        elif model_name in ["GPTNeo", "gptneo"]:
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-1.3B").to(self.device)
            
        elif model_name in ["llama2" ,"llama"]:
            self.model_dir = full_path(f"/.data/sharded_llama2/llama2_shard_size_1GB")
            self.model, self.tokenizer = llama2.get_llama_model_and_tokenizer(self.model_dir, hf_token=hf_auth, device=self.device)
            
        elif model_name == "chatgpt" or model_name=="gpt3":
            openai.api_key = keys["gpt3"]
            
        else:
            self.is_valid_model = False
            ValueError(f"Unknown model name : {model_name}")

class EmbeddingGenerator(ModelLoader):
    def __init__(self,model_name:str,default_gpu="cuda",
                 gpt3Model:str= "text-embedding-ada-002") -> None:
        super().__init__(model_name,default_gpu)
        self.device=default_gpu
        self.model_name = model_name
        if self.model_name in ["gpt3", "chatgpt"]:
            self.gpt3model = gpt3Model
        

    def generate_embedding(self,sentence: Union[str, List[str]]):
        if not self.is_valid_model:
            print(f"Error: Unknown model name '{self.model_name}'. Embedding generation aborted. ")
            return None

        
        if self.model_name in  ["llama2" , "llama"]:
            sentence_embeddings = llama2.llama2(sentence, self.model, self.tokenizer, device=self.device)
        
        elif self.model_name in  ["Bloom" , "bloom" ,"GPTNeo" , "gptneo"]:
            sent_encode = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device=self.device)
            
            with torch.no_grad():
                model_embedding = self.model(**sent_encode)
                
            sentence_embeddings = self._mean_pooling(model_embedding, sent_encode['attention_mask'])
        
        elif self.model_name in  ["chatgpt", "gpt3"]:
            sentence_embeddings=[]
            embeddings = openai.Embedding.create(
                model = self.gpt3model,
                input = sentence
                )
            for i in range(len(sentence)):
                sentence_embeddings.append(embeddings["data"][i]["embedding"])
                
        
        return sentence_embeddings
       
    
    #Mean Pooling - Take attention mask into account for correct averaging
    def _mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Usage
if __name__ == "__main__":
    
    device = torch.device('cuda:3' if torch.cuda.is_available() else "cpu")
    
    sentence1='this is testing'
    sentence2="this is not a testing"
    sentence3= "this is just a random sentence"
    sentence4 = " this is another sentence"
    sentence5="This sentence is longer than all other sentence"
    sent = [sentence1,sentence2,sentence3,sentence4,sentence5]
    batch = sent * 10
    generator = EmbeddingGenerator("gpt3",default_gpu=device)
    embedding = generator.generate_embedding(batch)

    print(embedding)
    print(embedding.shape)
    print(embedding[0].shape)