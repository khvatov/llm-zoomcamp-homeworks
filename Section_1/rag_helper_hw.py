from typing import override
from rag_helper import RAGBase

class RAGBaseHW1(RAGBase):

    @override
    def search(self, query, num_results=5):
        boost_dict={"content": 3.0, "filename": 0.5}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict
        )

    @override
    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc['filename'])
            lines.append('content: ' + doc['content'])
            lines.append('')

        return '\n'.join(lines).strip()

    @override
    def llm(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )
        return response

    @override
    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer