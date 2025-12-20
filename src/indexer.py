"""
인덱서 모듈
Inverted Index 구축 담당
"""
import pickle
import math
from collections import defaultdict
from tqdm import tqdm

from .tokenizer import Tokenizer


class InvertedIndex:
    """
    Inverted Index 구현
    
    구조:
    - posting_list: {term: [(doc_id, term_freq), ...]}
    - doc_len: {doc_id: 문서길이}
    - doc_store: {doc_id: 원본텍스트}
    """
    
    def __init__(self):
        self.posting_list = defaultdict(list)
        self.doc_freq = defaultdict(int)  # 각 term이 등장한 문서 수
        self.doc_len = {}
        self.doc_store = {}
        self.total_docs = 0
        self.avg_doc_len = 0
        
        self.tokenizer = Tokenizer()
    
    def add_document(self, doc_id, text):
        """문서 하나를 인덱스에 추가"""
        self.doc_store[doc_id] = text
        tokens = self.tokenizer.tokenize(text)
        self.doc_len[doc_id] = len(tokens)
        
        # term frequency 계산
        tf_dict = defaultdict(int)
        for token in tokens:
            tf_dict[token] += 1
        
        # posting list에 추가
        for term, tf in tf_dict.items():
            self.posting_list[term].append((doc_id, tf))
            self.doc_freq[term] += 1
    
    def build_from_file(self, filepath):
        """TSV 파일에서 인덱스 구축"""
        print(f"[Indexer] Loading documents from {filepath}")
        
        docs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            header = next(f)  # 헤더 스킵
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    docs.append((parts[0], parts[1]))
        
        print(f"[Indexer] Building index for {len(docs)} documents...")
        
        total_len = 0
        for doc_id, text in tqdm(docs, desc="Indexing"):
            self.add_document(doc_id, text)
            total_len += self.doc_len[doc_id]
        
        self.total_docs = len(docs)
        self.avg_doc_len = total_len / self.total_docs if self.total_docs > 0 else 0
        
        print(f"[Indexer] Done. {self.total_docs} docs, {len(self.posting_list)} unique terms")
    
    def get_posting(self, term):
        """term의 posting list 반환"""
        return self.posting_list.get(term, [])
    
    def get_doc_freq(self, term):
        """term의 document frequency 반환"""
        return self.doc_freq.get(term, 0)
    
    def get_document(self, doc_id):
        """원본 문서 반환"""
        return self.doc_store.get(doc_id, "")
    
    def save(self, path):
        """인덱스를 파일로 저장"""
        data = {
            'posting_list': dict(self.posting_list),
            'doc_freq': dict(self.doc_freq),
            'doc_len': self.doc_len,
            'doc_store': self.doc_store,
            'total_docs': self.total_docs,
            'avg_doc_len': self.avg_doc_len
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[Indexer] Index saved to {path}")
    
    def load(self, path):
        """파일에서 인덱스 로드"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.posting_list = defaultdict(list, data['posting_list'])
        self.doc_freq = defaultdict(int, data['doc_freq'])
        self.doc_len = data['doc_len']
        self.doc_store = data['doc_store']
        self.total_docs = data['total_docs']
        self.avg_doc_len = data['avg_doc_len']
        
        print(f"[Indexer] Loaded index: {self.total_docs} docs")
