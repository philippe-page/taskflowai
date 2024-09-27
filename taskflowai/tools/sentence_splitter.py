from typing import List
from sentence_splitter import SentenceSplitter

class TextSplitter:
    def __init__(self, language: str = 'en'):
        self.splitter = SentenceSplitter(language)

    def split_text_by_sentences(self, text: str, chunk_size: int = 5, overlap: int = 1) -> List[str]:
        """
        Split the text into chunks of sentences with overlap.
        
        :param text: The input text to split.
        :param chunk_size: The number of sentences per chunk.
        :param overlap: The number of sentences to overlap between chunks.
        :return: A list of text chunks.
        """
        sentences = self.splitter.split(text)
        chunks = []
        
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk = ' '.join(sentences[i:i + chunk_size])
            chunks.append(chunk.strip())
        
        print(f"Created {len(chunks)} chunks with {chunk_size} sentences each and {overlap} sentence overlap")
        return chunks
