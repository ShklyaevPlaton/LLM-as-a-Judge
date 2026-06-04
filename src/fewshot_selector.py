from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

class FewShotSelector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.examples = []
        self.tfidf_matrix = None
    
    def fit(self, df):
        self.examples = df.to_dict('records')
        dialogs = df['dialog'].tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(dialogs)
        print(f"закодировано примеров: {len(self.examples)}")
    
    def select(self, dialog, k=3):
        query_vec = self.vectorizer.transform([dialog])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.examples[i] for i in top_indices]


if __name__ == "__main__":
    df = pd.read_csv("data/processed/synthetic_dataset.csv")
    
    selector = FewShotSelector()
    selector.fit(df.head(100))
    
    test_dialog = df.iloc[0]['dialog']
    examples = selector.select(test_dialog, k=2)
    
    for i, ex in enumerate(examples):
        print(f"\nПример {i+1} : {ex['label']}")
        print(f"Диалог: {ex['dialog'][:150]}...")