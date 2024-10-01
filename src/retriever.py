from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

class retriver:
    def __init__(self):
        self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.qdrant_client = QdrantClient(url="http://localhost:6333")
        self.collection_name = "collection_bm25"
        self.query = ""

    def create_sparse_vector(self, text):
        """
        Create a sparse vector from the text using BM25.
        """
        model = SparseTextEmbedding(model_name="Qdrant/bm25")
        embeddings = list(model.embed(text))[0]

        sparse_vector = models.SparseVector(
            indices=embeddings.indices.tolist(),
            values=embeddings.values.tolist()
        )
        return sparse_vector
    
    def get_dense_embedding(self, text):
        """
        Get dense embedding for the given text using BERT-based model.
        """
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embedding = model.encode(text).tolist()
        return embedding  # Convert to list
    
    def hybrid_search(self, query: str):
        """
        Perform a hybrid search using both sparse and dense embeddings with RRF fusion.
        
        Args:
            query (str): The search query string.

        Returns:
            List[str]: A list of document texts based on the query.
        """
        # Generate sparse and dense embeddings for the query
        sparse_query = self.create_sparse_vector(query)
        dense_query = self.get_dense_embedding(query)
        
        # Perform hybrid search with RRF fusion strategy
        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=sparse_query,  # Sparse vector query
                    using="sparse",
                    limit=3
                ),
                models.Prefetch(
                    query=dense_query,  # Dense vector query
                    using="dense",
                    limit=3
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF)  # Rank Reciprocal Fusion
        )
        
        # Extract and return document texts from search results
        documents = [point.payload['text'] for point in search_results.points]
        print(f"Total documents retrieved: {len(documents)}")

        return documents

# Usage Example
if __name__ == '__main__':
    search = retriver()
    query = "Can you explain the objective of sustainable development?"
    results = search.hybrid_search(query)
    for doc in results:
        print(doc)
