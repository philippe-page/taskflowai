import os
from typing import List
import numpy as np
from dotenv import load_dotenv
from .embedding_tools import EmbeddingsTools
import igraph as ig
import leidenalg as la
from sentence_splitter import SentenceSplitter
load_dotenv()

class SemanticSplitter:
    def __init__(self, embedding_provider: str = "openai", embedding_model: str = "text-embedding-3-small"):
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.splitter = SentenceSplitter(language='en')

    @staticmethod
    def chunk_text(text: str, resolution: float = 0.35, similarity_threshold: float = 0.5, rearrange: bool = False, 
                   embedding_provider: str = "openai", embedding_model: str = "text-embedding-3-small") -> List[str]:
        splitter = SemanticSplitter(embedding_provider, embedding_model)
        segments = splitter._create_sentence_segments(text)
        embeddings = splitter._embed_segments(segments)
        communities = splitter._detect_communities(embeddings, resolution, similarity_threshold)
        chunks = splitter._create_chunks_from_communities(segments, communities, rearrange)
        
        print(f"Created {len(chunks)} non-empty chunks")
        return chunks

    def _create_sentence_segments(self, text: str) -> List[str]:
        sentences = self.splitter.split(text)
        segments = [sentence.strip() for sentence in sentences]
        print(f"Created {len(segments)} segments")
        return segments

    def _embed_segments(self, segments: List[str]) -> np.ndarray:
        embeddings, _ = EmbeddingsTools.get_embeddings(segments, self.embedding_provider, self.embedding_model)
        return np.array(embeddings)

    def _detect_communities(self, embeddings: np.ndarray, resolution: float, similarity_threshold: float) -> List[int]:
        if embeddings.shape[0] < 2:
            return [0]
        
        G = self._create_similarity_graph(embeddings, similarity_threshold)
        
        partition = self._find_optimal_partition(G, resolution)
        
        communities = partition.membership
        
        num_communities = len(set(communities))
        print(f"Resolution: {resolution}, Similarity Threshold: {similarity_threshold}, Communities: {num_communities}")
        
        return communities

    def _create_chunks_from_communities(self, segments: List[str], communities: List[int], rearrange: bool) -> List[str]:
        if rearrange:
            # Group segments by community
            community_groups = {}
            for segment, community in zip(segments, communities):
                if community not in community_groups:
                    community_groups[community] = []
                community_groups[community].append(segment)
            
            # Create chunks from rearranged communities
            chunks = [' '.join(group).strip() for group in community_groups.values() if group]
        else:
            # Create chunks respecting original order
            chunks = []
            current_community = communities[0]
            current_chunk = []
            
            for segment, community in zip(segments, communities):
                if community != current_community:
                    chunks.append(' '.join(current_chunk).strip())
                    current_chunk = []
                    current_community = community
                current_chunk.append(segment)
            
            # Add the last chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk).strip())
        
        return [chunk for chunk in chunks if chunk]  # Remove any empty chunks

    def _identify_breakpoints(self, communities: List[int]) -> List[int]:
        breakpoints = []
        for i in range(1, len(communities)):
            if communities[i] != communities[i-1]:
                breakpoints.append(i)
        return breakpoints

    def _create_similarity_graph(self, embeddings: np.ndarray, similarity_threshold: float) -> ig.Graph:
        similarities = np.dot(embeddings, embeddings.T)
        np.fill_diagonal(similarities, 0)
        similarities = np.maximum(similarities, 0)
        similarities = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities))
        
        # Apply similarity threshold
        adjacency_matrix = (similarities >= similarity_threshold).astype(int)
        
        G = ig.Graph.Adjacency(adjacency_matrix.tolist())
        G.es['weight'] = similarities[np.where(adjacency_matrix)]
        return G

    def _find_optimal_partition(self, G: ig.Graph, resolution: float) -> la.VertexPartition:
        return la.find_partition(
            G, 
            la.CPMVertexPartition,
            weights='weight',
            resolution_parameter=resolution
        )

    def _split_oversized_communities(self, membership: List[int], max_size: int) -> List[int]:
        community_sizes = {}
        for comm in membership:
            community_sizes[comm] = community_sizes.get(comm, 0) + 1
        
        new_membership = []
        current_comm = max(membership) + 1
        for i, comm in enumerate(membership):
            if community_sizes[comm] > max_size:
                if i % max_size == 0:
                    current_comm += 1
                new_membership.append(current_comm)
            else:
                new_membership.append(comm)
        
        return new_membership
    
# Example usage
#text = "This is a test text to demonstrate the semantic splitter. It should be split into meaningful chunks based on the content and similarity threshold."

# Using OpenAI (default)
#chunks = SemanticSplitter.chunk_text(text)

# Using Cohere
#chunks = SemanticSplitter.chunk_text(text, embedding_provider="cohere", embedding_model="embed-english-v3.0")

# Using Mistral
#chunks = SemanticSplitter.chunk_text(text, embedding_provider="mistral", embedding_model="mistral-embed")