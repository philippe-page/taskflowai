from typing import List, Dict, Any, Optional
from ..tools import EmbeddingsTools
from ..tools import FAISSTools
import os
import json

class FaissKnowledgeBase:
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
        self.index_tool = None
        self.chunks = []

        try:
            import faiss
            import numpy as np
        except ModuleNotFoundError as e:
            raise ImportError(f"{e.name} is required for KnowledgeBase. Install with `pip install {e.name}`")

        self.faiss = faiss
        self.np = np

        if load_from_index:
            self.load_from_index(load_from_index)
        elif chunks:
            self.initialize_from_chunks(chunks, **kwargs)
            if save_to_filepath:
                self.save_index(save_to_filepath)
        else:
            self.initialize_empty(**kwargs)
    
    def initialize_from_chunks(self, chunks: List[str],
                               save_to_filepath: Optional[str] = None,
                               **kwargs) -> None:
        try:
            self.chunks = chunks

            embeddings, _ = EmbeddingsTools.get_embeddings(chunks, provider=self.embedding_provider, model=self.embedding_model)

            self.index_tool = FAISSTools(dimension=len(embeddings[0]), metric=kwargs.get("metric", "IP"))
            self.index_tool.create_index()
            self.index_tool.set_embedding_info(self.embedding_provider, self.embedding_model)

            np_vectors = self.np.array(embeddings).astype('float32')
            self.index_tool.add_vectors(np_vectors)

            self.index_tool.set_metadata('chunks', chunks)

            if save_to_filepath:
                self.index_tool.save_index(save_to_filepath)
        except Exception as e:
            raise Exception(f"Error initializing knowledgebase: {str(e)}")
    
    def initialize_empty(self, **kwargs):
        try:
            dimension = EmbeddingsTools.get_model_dimension(self.embedding_provider, self.embedding_model)
            if not dimension:
                raise ValueError(f"Unsupported embedding model: {self.embedding_model} for provider: {self.embedding_provider}")
            
            self.index_tool = FAISSTools(dimension=dimension, metric=kwargs.get("metric", "IP"))
            self.index_tool.create_index()
            self.index_tool.set_embedding_info(self.embedding_provider, self.embedding_model)
            self.index_tool.set_metadata('chunks', [])
        except Exception as e:
            raise Exception(f"Error initializing empty knowledgebase: {str(e)}")

    def load_from_index(self, index_path: str) -> None:
        try:
            index_file = index_path
            metadata_file = f"{index_path}.metadata"

            if not os.path.exists(index_file):
                raise FileNotFoundError(f"Index file not found at {index_file}")
            
            if not os.path.exists(metadata_file):
                raise FileNotFoundError(f"Metadata file not found at {metadata_file}")

            self.index_tool = FAISSTools(dimension=1)  # Dimension will be updated when loading
            self.index_tool.load_index(index_file)

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            self.chunks = metadata.get('chunks')
            if self.chunks is None:
                raise ValueError(f"No chunks found in metadata for '{self.kb_name}'")

            self.embedding_provider = self.index_tool.embedding_provider
            self.embedding_model = self.index_tool.embedding_model

            if not self.embedding_provider or not self.embedding_model:
                raise ValueError(f"No embedding provider or model found in metadata for '{self.kb_name}'")

        except Exception as e:
            raise Exception(f"Error loading knowledgebase: {str(e)}")
    
    def query(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []
        try:
            if self.index_tool is None:
                raise ValueError(f"Knowledgebase '{self.kb_name}' is not initialized. Please initialize or load the knowledgebase first.")

            query_embedding, _ = EmbeddingsTools.get_embeddings([query], provider=self.embedding_provider, model=self.embedding_model)

            query_vector = self.np.array(query_embedding).astype('float32')
            distances, indices = self.index_tool.search_vectors(
                query_vectors=query_vector, top_k=top_k)

            formatted_results = []
            for idx, dist in zip(indices[0], distances[0]):
                chunk_index = int(idx)
                formatted_results.append({
                    "id": str(chunk_index),
                    "score": round(float(dist), 4),
                    "content": self.chunks[chunk_index]
                })

            return formatted_results

        except Exception as e:
            raise Exception(f"Error querying knowledgebase: {str(e)}")
    
    def add_memory(self, memory: str) -> Dict[str, Any]:
        try:
            if self.index_tool is None:
                raise ValueError(f"Knowledgebase '{self.kb_name}' is not initialized. Please initialize or load the knowledgebase first.")

            new_embedding, _ = EmbeddingsTools.get_embeddings([memory], provider=self.embedding_provider, model=self.embedding_model)
            new_vector = self.np.array(new_embedding).astype('float32')

            self.index_tool.add_vectors(new_vector)
            self.chunks.append(memory)
            self.index_tool.set_metadata('chunks', self.chunks)

            return {"success": True, "message": "Memory added successfully"}

        except Exception as e:
            error_message = f"Error adding memory to knowledgebase: {str(e)}"
            print(error_message)
            return {"success": False, "message": error_message}
    
    def save_index(self, save_to_filepath: Optional[str] = None) -> None:
        try:
            if self.index_tool is None:
                raise ValueError(f"Knowledgebase '{self.kb_name}' is not initialized or loaded.")
            
            filepath = save_to_filepath or self.index_tool.last_save_path
            if not filepath:
                raise ValueError("No filepath provided and no known filepath for the index.")

            # Create a 'faiss_indexes' folder in the same directory as the filepath
            indexes_dir = os.path.join(os.path.dirname(filepath), 'faiss_indexes')
            os.makedirs(indexes_dir, exist_ok=True)

            # Update the filepath to save in the 'faiss_indexes' folder
            filename = os.path.basename(filepath)
            new_filepath = os.path.join(indexes_dir, filename)

            self.index_tool.save_index(new_filepath)
            self.index_tool.last_save_path = new_filepath
        except Exception as e:
            raise Exception(f"Error saving knowledgebase index: {str(e)}")