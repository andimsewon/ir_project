"""
토크나이저 모듈
텍스트를 토큰으로 분리하는 기능
"""
import re

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "what", "which", "who", "whom", "where", "when", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very"
}


class Tokenizer:
    """영어 텍스트 토크나이저"""
    
    def __init__(self, use_lowercase=True, use_stopwords=True, min_len=2):
        self.use_lowercase = use_lowercase
        self.use_stopwords = use_stopwords
        self.min_len = min_len
        self._pattern = re.compile(r'[a-zA-Z0-9]+')
    
    def tokenize(self, text):
        """텍스트를 토큰 리스트로 변환"""
        if self.use_lowercase:
            text = text.lower()
        
        tokens = self._pattern.findall(text)
        
        if self.use_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        
        tokens = [t for t in tokens if len(t) >= self.min_len]
        
        return tokens
