from typing import Union, List, Generator
from numpy import ndarray
import gensim.downloader
from gensim.utils import tokenize


class SentenceEmbedding:
    """ Utility class that embeds text using pre-trained word embeddings. """
    def __init__(self, embedding: str) -> None:
        """ Constructor.

        :param embedding: name of a Gensim word embedding
        :type  embedding: str

        Note: embedding will be downloaded to machine (see Gensim docs).
        """
        self.name = embedding
        self.embedding = gensim.downloader.load(embedding)

    def __repr__(self) -> str:
        """ Object representation.

        :return: how to re-create the instance that calls this method
        :rtype:  str
        """
        return f'SentenceEmbedding({self.name})'

    def _embed_single(self, sentence: str) -> Generator[ndarray, None, None]:
        """ Tokenize and embed a sentence. Skips words not in vocabulary.

        :param sentence: English text
        :type  sentence: str
        :yield:          word embeddings one word at a time
        :rtype:          Generator[ndarray, None, None]
        """
        for token in tokenize(sentence, lower=True):
            try:
                wordvec = self.embedding[token]
            except KeyError:
                continue
            yield wordvec

    def embed(self, sentences: List[str], full=False) -> \
            List[Union[Generator[ndarray, None, None], List[ndarray]]]:
        """ Tokenize and embed multiple sentences. Skips words not in vocabulary.

        :param sentences: English text
        :type  sentences: List[str]
        :param full:      whether to return generators or fully-embedded text,
                          defaults to False
        :type  full:      bool, optional
        :return:          embedded text
        :rtype:           List[Union[
                              Generator[ndarray, None, None],
                              List[ndarray]
                          ]]
        """
        embedded = [self._embed_single(s) for s in sentences]
        if full:
            embedded = [list(s) for s in embedded]
        return embedded
