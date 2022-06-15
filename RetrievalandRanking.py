"""
Measuring Text Similarity Using BERT

Author: Nguyen Thi Hong Phuc
Tutorial: https://www.analyticsvidhya.com/blog/2021/05/measuring-text-similarity-using-bert/
"""
sentences = [
    "Three years later, the coffin was still full of Jello.",
    "He found a leprechaun in his walnut shell.",
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
    "The person box was packed with jelly many dozens of months later.",
]

MAX_LENGTH = 128
LAST_HIDDEN_STATE = 768
TOP_K = 1

import numpy as np 
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

"""
Tokenizing Sentences
"""
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = {'input_ids': [], 'attention_mask': [] }

def tokenize_sentences(sentences):
    for sentence in sentences:
        update_tokens = tokenizer.encode_plus(sentence, max_length=MAX_LENGTH,
                                                truncation=True, padding='max_length', 
                                                return_tensors='pt'
        )
        tokens['input_ids'].append(update_tokens['input_ids'][0])
        tokens['attention_mask'].append(update_tokens['attention_mask'][0])


    # reformat list of tensors to single tensor: (seg_len, max_len)
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

    return tokens

tokens = tokenize_sentences(sentences)

"""
Retrieval Sentence - BERT_model
Input: dictionary
Return:
    if return_dict=False: specific output objects (seg_num, max_leng, hidden_state_features_num)
    if return_dict=True: Tuple (seg_num, max_leng, hidden_state_features_num)

""" 
def generate_embedding(tokens, init_wts=False):
    model = AutoModel.from_pretrained("bert-base-uncased")
    if init_wts: 
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    outputs = model(**tokens, return_dict=True)
    embeddings = outputs.last_hidden_state
    pooler_output = outputs.pooler_output

    return embeddings, pooler_output

def create_msk_padding(tokens, embeddings):
    attn_msk = tokens['attention_mask']
    attn_msk = attn_msk.unsqueeze(-1).expand(embeddings.size()).float()   
    return attn_msk 

"""
Mean Pooling: average the meaning of sentences
Return: transform last_hidden_states tensor into desire vector
(seq_num, max_len, last_hidden_states) -> (seq_num, last_hidden_states)
"""
def mean_pooling(tokens, eps=1e-9):
    embeddings, _  = generate_embedding(tokens)
    attn_msks      = create_msk_padding(tokens, embeddings)
    embedding_msks = embeddings * attn_msks 

    sum_embedding = torch.sum(embedding_msks, 1)
    sum_mask      = torch.clamp(attn_msks.sum(1), min=eps)
    mean_pooling   = sum_embedding / sum_mask
    return mean_pooling

mean_values = mean_pooling(tokens, eps=1e-9).detach().numpy()
# print('mean_pooling_vector:', mean_values.shape)

"""
    Similarity matrix: Cosine_similarity
"""
from sklearn.metrics.pairwise import cosine_similarity
output = cosine_similarity([mean_values[0]], mean_values[1:])
idc    = np.argmax(output) + 1
print("Cosine_similarity:", sentences[idc])

"""
    Similarity matrix: FAISS
"""
import faiss      

class RetrievalandRanking(nn.Module):
    def __init__(self, mean_embedding, top_k):
        super().__init__()
        self.query_vec =  mean_embedding[0][np.newaxis]
        self.lib_vec   =  mean_embedding[1:]
        self.top_k = top_k

    def check_similarity(self):
        self.index = faiss.IndexFlatL2(LAST_HIDDEN_STATE)   
        self.index.add(self.lib_vec)                             
        dis, idc = self.index.search(self.query_vec, k=self.top_k)
        return dis, idc

    def forward(self, sentences):
        top_dis, top_idc = self.check_similarity()
        top_sentence = [sentences[idc+1] for idc in top_idc[0]]
        return top_sentence

    def extra_repr(self):
        return f"top_k={self.top_k}, index_trained={self.index.is_trained}, index_ntotal={self.index.ntotal}"

result = RetrievalandRanking(mean_embedding=mean_values, top_k=TOP_K)
top_sentences = result(sentences)
print("FAISS_similarity :", top_sentences[0])


