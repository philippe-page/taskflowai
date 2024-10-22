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

from typing import List
from sentence_splitter import SentenceSplitter

class TextSplitter:
    @staticmethod
    def split_text_by_sentences(text: str, chunk_size: int = 5, overlap: int = 1, language: str = 'en') -> List[str]:
        """
        Split the text into chunks of sentences with overlap.
        
        :param text: The input text to split.
        :param chunk_size: The number of sentences per chunk.
        :param overlap: The number of sentences to overlap between chunks.
        :param language: The language of the text (default: 'en').
        :return: A list of text chunks.
        """
        splitter = SentenceSplitter(language)
        sentences = splitter.split(text)
        chunks = []
        
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk = ' '.join(sentences[i:i + chunk_size])
            chunks.append(chunk.strip())
        
        print(f"Created {len(chunks)} chunks with {chunk_size} sentences each and {overlap} sentence overlap")
        return chunks
