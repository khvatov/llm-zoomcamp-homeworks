from typing import override
import numpy as np
from rag_helper import RAGBase


class RAGVector(RAGBase):
    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder


    @override
    def search(self, query, num_results=5):
        query_vector = np.array(self.llm_client.embeddings.create(input=[query], model=self.embedder).data[0].embedding)
        filter_dict = {'course': self.course}
        results = self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict=filter_dict
        )
        return results