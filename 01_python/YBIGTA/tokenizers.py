import numpy as np
from typing import Optional, Union, List
from collections import Counter, defaultdict
import re

class BPETokenizer:
    def __init__(self, corpus):
        self.corpus = []
        self.word_freqs = defaultdict(int)
        self.splits = {}
        self.merges = {}
        self.token = []
        
        if isinstance(corpus, str):
            pre_corpus = re.split(' ', (re.sub(r'\s+', "</w> ", corpus)))
            for word in pre_corpus:
                self.corpus.extend(re.split('([.,\[\]?!"])', word))
            self.corpus = [item for item in self.corpus if item]
        
        elif all(isinstance(item, str) for item in corpus):
            pre_corpus = []
            for sentence in corpus:
                pre_corpus.extend(re.split(' ', (re.sub(r'\s+', "</w> ", sentence))))
                
            for word in pre_corpus:
                self.corpus.extend(re.split('([.,\[\]?!"])', word))
                
            self.corpus = [item for item in self.corpus if item]
            
    def add_corpus(self, corpus):
        if isinstance(corpus, str):
            pre_corpus = re.split(' ', (re.sub(r'\s+', "</w> ", corpus)))
            for word in pre_corpus:
                self.corpus.extend(re.split('([.,\[\]?!"])', word))
            self.corpus = [item for item in self.corpus if item]
        
        elif all(isinstance(item, str) for item in corpus):
            pre_corpus = []
            for sentence in corpus:
                pre_corpus.extend(re.split(' ', (re.sub(r'\s+', "</w> ", sentence))))
                
            for word in pre_corpus:
                self.corpus.extend(re.split('([.,\[\]?!"])', word))
                
            self.corpus = [item for item in self.corpus if item]
                
        
    
    
    def train(self, n_iter):
        for word in self.corpus:
            self.word_freqs[word] += 1
        
        character = []
        for word in self.word_freqs.keys():
            for letter in word:
                if letter not in character:
                    character.append(letter)
        
        character.sort()
        self.token = ["</w>"] + character.copy()
        
        self.splits = {word: [letter for letter in word] for word in self.word_freqs.keys()}
        
        for i in range(n_iter):
            print(i)
            max_feq = None
            max_pair = ''
            pair_cnt = self.cnt_feq()
            for pair, cnt in pair_cnt.items():
                if max_feq is None or max_feq < cnt:
                    max_feq = cnt
                    max_pair = pair
            
            self.splits = self.merge_max(*max_pair)
            self.merges[max_pair] = max_pair[0] + max_pair[1]
            self.token.append(max_pair[0] + max_pair[1])
        
        return self.merges
            
    
    
    def cnt_feq(self):
        pair_cnt = defaultdict(int)
        for word, feq in self.word_freqs.items():
            sp_word = self.splits[word]
            if len(sp_word) == 1:
                continue
            for a in range(len(sp_word) - 1):
                pair_cnt[(sp_word[a],  sp_word[a+1])] += feq
            
        return pair_cnt
                    
    
    def merge_max(self, x, y):
        for word in self.word_freqs:
            sp_word = self.splits[word]
            if len(sp_word) == 1:
                continue
            a = 0
            while a < len(sp_word) - 1:
                if sp_word[a] == x and sp_word[a+1] == y:
                    sp_word = sp_word[:a] + [x+y] + sp_word[a+2:]
                a += 1
            self.splits[word] = sp_word
        return self.splits
    
    def help_tokenize(self, splits_corpus):
        for pair, merge in self.merges.items():
            for index, split in enumerate(splits_corpus):
                a = 0
                while a < len(split) - 1:
                    if split[a] == pair[0] and split[a + 1] == pair[1]:
                        split = split[:a] + [merge] + split[a+2:]
                    a += 1
                splits_corpus[index] = split
        result = sum(splits_corpus, [])
        return result
    
    def tokenize(self, new_corpus):
        token_corpus = []
        print(self.token[0])
        if isinstance(new_corpus, str):
            pre_corpus = re.split(' ', (re.sub(r'\s+', "</w> ", new_corpus)))
            for word in pre_corpus:
                token_corpus.extend(re.split('([.,\[\]?!"])', word))
            
            splits_corpus = [[l for l in word] for word in token_corpus]
            result = self.help_tokenize(splits_corpus)
            result_id = []
            for word in result:
                for index, tok in enumerate(self.token):
                    if word == tok:
                        result_id.append(index)
            
            return result_id
        
        elif all(isinstance(item, str) for item in new_corpus):
            result_id = []
            for sentence in new_corpus:
                pre_corpus = re.split(' ', (re.sub(r'\s+', "</w> ", sentence)))
                for word in pre_corpus:
                    token_corpus.extend(re.split('([.,\[\]?!"])', word))
                
                splits_corpus = [[l for l in word] for word in token_corpus]
                result = self.help_tokenize(splits_corpus)
                sentence_id = []
                for word in result:
                    for index, tok in enumerate(self.token):
                        if word == tok:
                            sentence_id.append(index)
                result_id.append(sentence_id)
            return result_id
    
    def print_token(self):
        list1 = [0, 300, 32, 334, 336, 330, 386, 36, 309, 7, 306, 67, 316, 361, 343, 359, 66, 85, 76, 308, 67, 301, 375, 76, 337, 302, 89, 318, 0, 300, 394, 83, 368, 85, 84, 344, 313, 78, 69, 317, 73, 306, 32, 334, 336, 330, 386, 52, 82, 313, 323, 66, 301, 80, 79, 330, 84, 301, 323, 310, 37, 50, 0, 300, 80, 307, 83, 309, 78, 344, 12, 0, 300, 328, 68, 309, 7, 306, 330, 301, 346, 77, 73, 83, 320, 80, 320, 83, 379, 308, 89, 303, 325, 32, 334, 336, 330, 386, 33, 302, 322, 76, 351, 306, 320, 83, 312, 84, 12, 0, 300, 67, 316, 361, 310, 72, 397, 80, 356, 354, 382, 77, 304, 73, 329, 82, 332, 346, 333, 67, 316, 76]
        decode = []
        for l in list1:
            decode.append(self.token[l])
        print(decode)

class WordTokenizer:
    pass