from typing import List, Dict, Any, Optional
from ..tools import EmbeddingsTools
from ..tools import PineconeTools
import os
import json

class PineconeKnowledgeBase:
    def __init__(self, kb_name: str = "default",
                 embedding_provider: str = "openai",
                 embedding_model: str = "text-embedding-3-small",
                 load_from_index: Optional[str] = None,
                 chunks: Optional[List[str]] = None,
                 save_to_filepath: Optional[str] = None,
                 **kwargs):
        self.kb_name = kb_name
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.pinecone_tools = PineconeTools()
        self.chunks = []
        
        if load_from_index:
            self.load_from_index(load_from_index)
        elif chunks:
            self.initialize_from_chunks(chunks, save_to_filepath, **kwargs)
        else:
            self.initialize_empty(**kwargs)
    
    def initialize_from_chunks(self, chunks: List[str],
                               save_to_filepath: Optional[str] = None,
                               **kwargs) -> None:
        try:
            self.chunks = chunks

            embeddings, _ = EmbeddingsTools.get_embeddings(chunks, provider=self.embedding_provider, model=self.embedding_model)

            dimension = len(embeddings[0])
            metric = kwargs.get("metric", "cosine")

            # Create Pinecone index if it doesn't exist
            if self.kb_name not in self.pinecone_tools.list_indexes():
                self.pinecone_tools.create_index(self.kb_name, dimension, metric)

            # Prepare vectors for upsert
            vectors = [
                {"id": str(i), "values": embedding, "metadata": {"text": chunk}}
                for i, (embedding, chunk) in enumerate(zip(embeddings, chunks))
            ]

            # Upsert vectors to Pinecone
            self.pinecone_tools.upsert_vectors(self.kb_name, vectors)

            if save_to_filepath:
                self.save_metadata(save_to_filepath)

        except Exception as e:
            raise Exception(f"Error initializing knowledgebase: {str(e)}")
    
    def initialize_empty(self, **kwargs):
        try:
            dimension = EmbeddingsTools.get_model_dimension(self.embedding_provider, self.embedding_model)
            if not dimension:
                raise ValueError(f"Unsupported embedding model: {self.embedding_model} for provider: {self.embedding_provider}")
            
            metric = kwargs.get("metric", "cosine")
            
            # Create empty Pinecone index if it doesn't exist
            if self.kb_name not in self.pinecone_tools.list_indexes():
                self.pinecone_tools.create_index(self.kb_name, dimension, metric)

        except Exception as e:
            raise Exception(f"Error initializing empty knowledgebase: {str(e)}")

    def load_from_index(self, index_path: str) -> None:
        try:
            metadata_file = f"{index_path}.metadata"

            if not os.path.exists(metadata_file):
                raise FileNotFoundError(f"Metadata file not found at {metadata_file}")

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            self.chunks = metadata.get('chunks')
            if self.chunks is None:
                raise ValueError(f"No chunks found in metadata for '{self.kb_name}'")

            self.embedding_provider = metadata.get('embedding_provider')
            self.embedding_model = metadata.get('embedding_model')

            if not self.embedding_provider or not self.embedding_model:
                raise ValueError(f"No embedding provider or model found in metadata for '{self.kb_name}'")

        except Exception as e:
            raise Exception(f"Error loading knowledgebase: {str(e)}")
    
    def query(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        try:
            query_embedding, _ = EmbeddingsTools.get_embeddings([query], provider=self.embedding_provider, model=self.embedding_model)

            query_vector = PineconeTools.normalize_vector(query_embedding[0])
            results = self.pinecone_tools.query_index(self.kb_name, query_vector, top_k=top_k)

            formatted_results = []
            for match in results.get('matches', []):
                formatted_results.append({
                    "id": match['id'],
                    "score": match['score'],
                    "content": match['metadata']['text']
                })

            return formatted_results

        except Exception as e:
            raise Exception(f"Error querying knowledgebase: {str(e)}")
    
    def add_memory(self, memory: str) -> Dict[str, Any]:
        try:
            new_embedding, _ = EmbeddingsTools.get_embeddings([memory], provider=self.embedding_provider, model=self.embedding_model)
            new_vector = PineconeTools.normalize_vector(new_embedding[0])

            new_id = str(len(self.chunks))
            vector = {
                "id": new_id,
                "values": new_vector,
                "metadata": {"text": memory}
            }

            self.pinecone_tools.upsert_vectors(self.kb_name, [vector])
            self.chunks.append(memory)

            return {"success": True, "message": "Memory added successfully"}

        except Exception as e:
            error_message = f"Error adding memory to knowledgebase: {str(e)}"
            print(error_message)
            return {"success": False, "message": error_message}
    
    def save_metadata(self, save_to_filepath: str) -> None:
        try:
            metadata = {
                'chunks': self.chunks,
                'embedding_provider': self.embedding_provider,
                'embedding_model': self.embedding_model
            }

            directory = os.path.dirname(save_to_filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            with open(f"{save_to_filepath}.metadata", 'w') as f:
                json.dump(metadata, f)

        except Exception as e:
            raise Exception(f"Error saving knowledgebase metadata: {str(e)}")

    def save_index(self, save_to_filepath: Optional[str] = None) -> None:
        try:
            # Pinecone indexes are saved automatically in the cloud,
            # so we only need to save the metadata
            if save_to_filepath:
                self.save_metadata(save_to_filepath)
            else:
                raise ValueError("No filepath provided for saving metadata.")

        except Exception as e:
            raise Exception(f"Error saving knowledgebase index: {str(e)}")