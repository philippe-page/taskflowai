# Copyright 2024 Philippe Page and TaskFlowAI Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2024 Philippe Page and TaskFlowAI Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Dict, Any, Optional
from ..tools import EmbeddingsTools
from ..tools import FAISSTools
import os
import json
import uuid

class FaissKnowledgeBase:
    """
    A knowledge base that uses FAISS for efficient similarity search.
    
    Parameters:
    - kb_name (str): The name of the knowledge base.
    - embedding_provider (str): The provider of the embeddings.
    - embedding_model (str): The model used for the embeddings.
    - load_from_index (str, optional): The path to load the index from.
    - chunks (List[str], optional): The chunks of text to initialize the knowledge base with.
    - save_to_filepath (str, optional): The path to save the index to.
    - **kwargs: Additional keyword arguments.

    Examples:
    >>> kb = FaissKnowledgeBase("default", "openai", "text-embedding-3-small")
    >>> kb = FaissKnowledgeBase("default", "openai", "text-embedding-3-small", load_from_index="index.faiss")
    >>> kb = FaissKnowledgeBase("default", "openai", "text-embedding-3-small", chunks=["chunk1", "chunk2"])
    >>> kb = FaissKnowledgeBase("default", "openai", "text-embedding-3-small", chunks=["chunk1", "chunk2"], save_to_filepath="index.faiss")
    """
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
        self.memories = {}  # Dictionary to store memories with their IDs
        self.save_filepath = save_to_filepath

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
        else:
            self.initialize_empty(**kwargs)

        if self.save_filepath:
            self.save_index(self.save_filepath)
            print(f"Index saved to {self.save_filepath}")
    
    def _format_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Helper method to format a memory with ID and metadata."""
        return {
            "id": str(len(self.memories) + 1),  # Simple incremental ID
            "content": content,
            "metadata": metadata or {}
        }
    
    def initialize_from_chunks(self, chunks: List[str],
                               **kwargs) -> None:
        try:
            formatted_chunks = [self._format_memory(chunk) for chunk in chunks]
            self.memories = {chunk['id']: chunk for chunk in formatted_chunks}
            chunk_texts = [chunk['content'] for chunk in formatted_chunks]

            embeddings, _ = EmbeddingsTools.get_embeddings(chunk_texts, provider=self.embedding_provider, model=self.embedding_model)

            self.index_tool = FAISSTools(dimension=len(embeddings[0]), metric=kwargs.get("metric", "IP"))
            self.index_tool.create_index()
            self.index_tool.set_embedding_info(self.embedding_provider, self.embedding_model)

            np_vectors = self.np.array(embeddings).astype('float32')
            self.index_tool.add_vectors(np_vectors)

            self.index_tool.set_metadata('memories', list(self.memories.values()))

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

            self.memories = {memory['id']: memory for memory in metadata.get('memories', [])}
            if not self.memories:
                raise ValueError(f"No memories found in metadata for '{self.kb_name}'")

            self.embedding_provider = self.index_tool.embedding_provider
            self.embedding_model = self.index_tool.embedding_model

            if not self.embedding_provider or not self.embedding_model:
                raise ValueError(f"No embedding provider or model found in metadata for '{self.kb_name}'")

        except Exception as e:
            raise Exception(f"Error loading knowledgebase: {str(e)}")
    
    def query(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """
        Query the knowledge base for the most relevant unique chunks.
        
        Args:
            query (str): The query to search for.
            top_k (int): The number of unique results to return.
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing the id, score, and content of a relevant unique chunk.
        """
        if not self.memories:
            return []
        try:
            if self.index_tool is None:
                raise ValueError(f"Knowledgebase '{self.kb_name}' is not initialized. Please initialize or load the knowledgebase first.")

            query_embedding, _ = EmbeddingsTools.get_embeddings([query], provider=self.embedding_provider, model=self.embedding_model)

            query_vector = self.np.array(query_embedding).astype('float32')
            
            # Increase the number of results to search for to ensure we have enough unique results
            distances, indices = self.index_tool.search_vectors(
                query_vectors=query_vector, top_k=top_k)

            formatted_results = []
            seen_contents = set()
            
            for idx, dist in zip(indices[0], distances[0]):
                memory = list(self.memories.values())[idx]
                content = memory['content']
                
                # Skip if we've already seen this content
                if content in seen_contents:
                    continue
                
                seen_contents.add(content)
                formatted_results.append({
                    "id": memory['id'],
                    "score": round(float(dist), 4),
                    "content": content
                })
                
                # Break if we have enough unique results
                if len(formatted_results) == top_k:
                    break

            return formatted_results

        except Exception as e:
            raise Exception(f"Error querying knowledgebase: {str(e)}")
    
    def add_memory(self, memory: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a memory to the knowledge base and save the updated index.
        
        Args:
            memory (str): The memory to add.
        
        Returns:
            Dict[str, Any]: A dictionary containing the success status and message.
        """
        try:
            if self.index_tool is None:
                raise ValueError(f"Knowledgebase '{self.kb_name}' is not initialized. Please initialize or load the knowledgebase first.")

            memory_obj = self._format_memory(memory, metadata)
            memory_id = memory_obj['id']

            new_embedding, _ = EmbeddingsTools.get_embeddings([memory], provider=self.embedding_provider, model=self.embedding_model)
            new_vector = self.np.array(new_embedding).astype('float32')

            self.index_tool.add_vectors(new_vector)
            self.memories[memory_id] = memory_obj
            self.index_tool.set_metadata('memories', list(self.memories.values()))

            if self.save_filepath:
                self.save_index(self.save_filepath)
            else:
                print("Warning: No save path set for the index. Changes are only in memory.")

            return {"success": True, "message": "Memory added successfully and index saved", "id": memory_id}
        except Exception as e:
            error_message = f"Error adding memory to knowledgebase: {str(e)}"
            print(error_message)
            return {"success": False, "message": error_message}
    
    def save_index(self, save_to_filepath: Optional[str] = None) -> None:
        try:
            if self.index_tool is None:
                raise ValueError(f"Knowledgebase '{self.kb_name}' is not initialized or loaded.")
            
            filepath = save_to_filepath or self.save_filepath
            if not filepath:
                raise ValueError("No filepath provided and no known filepath for the index.")

            self.index_tool.save_index(filepath)
            self.save_filepath = filepath
            #print(f"Index saved to {filepath}")
        except Exception as e:
            raise Exception(f"Error saving knowledgebase index: {str(e)}")
