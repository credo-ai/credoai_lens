import numpy as np

from credoai.modules.credo_module import CredoModule
from credoai.utils.common import NotRunError, wrap_list
from ._nlp_constants import (
    COMPETENCE, FAMILY, STEM, OUTSIDER,
    MALE, FEMALE
)
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_CATEGORIES = {'competence': COMPETENCE,
                      'family': FAMILY,
                      'STEM': STEM,
                      'outsider': OUTSIDER}

class NLPEmbeddingAnalyzer(CredoModule):
    """
    NLP embedding analyzer for Credo AI

    This module takes in a function that returns an embedding
    vector for a word. Using this function it calculates
    bias metrics similar to Bolukbasi et al. (2016). Simply put,
    the module calculates whether words associated whether group 
    category words (e.g. words associated with men vs women) 
    are "closer" to a set of comparison words (STEM careers). 

    Parameters
    ----------
    embedding_fun : callable
        function that takes in a string and outputs and one-dimensional embedding for the word
    """   
    def __init__(self,
             embedding_fun,
             ):      
        self.embedding_fun = embedding_fun
        self.group_embeddings = {}
        self.bias_categories = {}
        self.group_words = {}
        self.bias_category_embeddings = {}
        # set up group embeddings and bias categories
        # default to male and female groups
        self.set_group_embeddings()
        self.set_comparison_categories()

    def run(self, group1='male', group2='female'):
        """Evaluate bias between groups on the set comparison categories

        Bias will be calculated as group1-group2. That is, positive biases
        will indicate a closer relationship between the comparison categories
        and group1. Negative values indicate a closer relationship between
        the comparison categories and group2.

        Parameters
        ----------
        group1 : str
            string identifying one group from the group_words defined by 
            set_group_embeddings
        group2 : str
            string identifying comparison group from the group_words defined by 
            set_group_embeddings

        Returns
        -------
        [type]
            [description]
        """        
        # calculate biases
        biases = {}
        for category, word_embeddings in self.bias_category_embeddings.items():
            biases[category] = self._normalized_directional_bias(
                self.group_embeddings[group1], 
                self.group_embeddings[group2], 
                word_embeddings)
        self.results = biases
        return self

    def prepare_results(self):
        if self.results is not None:
            return self.results
        else:
            raise NotRunError(
                "Results not created yet. Call 'run' to create results"
            )
    
    def set_group_embeddings(self, group_words=None):
        """Set the group categories

        Each category (e.g. men) is defined by a set of words:
        {'male': ["he", "father", "son", ...],
         'female': ["she", "mother", "daughter", ...]}

        Parameters
        ----------
        group_words : dict, optional
            dictionary defining the associations between group categories
            and the words that define them. If no group_words are defined
            default male/female will be used., by default None
        """        
        if group_words is None:
            group_words = {'male': MALE, 'female': FEMALE}
        self.group_words = group_words
        self.group_embeddings = {k: self._embed_words(words)
                                for k, words in group_words.items()}
    
    def set_comparison_categories(self, include_default=True, custom_categories=None):
        """Set the comparison categories

        Each category (e.g. STEM) is defined by a set of words:
        {'STEM': ["computer", "math", "programming", ...],
         'COMPETENCE': ["precocious", "resourceful", "astute", ...]}

        Parameters
        ----------
        include_default : bool, optional
            includes a set of Credo AI defined default comparison categories, by default True
        custom_categories : dictionary, optional
            dictionary defining the associations between comparison categories
            and the words that define them, by default None
        """        
        # determine categories
        bias_categories = {}
        if include_default:
            bias_categories.update(DEFAULT_CATEGORIES)
        if custom_categories is not None:
            bias_categories.update(custom_categories)
        self.bias_categories = bias_categories
        self.bias_category_embeddings = {k: self._embed_words(v) 
                                         for k, v in self.bias_categories.items()}
    
    def get_group_words(self):
        return self.group_words
    
    def get_bias_categories(self):
        return self.bias_categories


    def _embed_words(self, words):
        words = wrap_list(words)
        tmp = []
        for word in words:
            emb = self.embedding_fun(word)
            if emb is not None:
                tmp.append(emb)
        mat = np.vstack(tmp)
        # normalize to unit norm
        return mat/np.linalg.norm(mat, axis=1)[:, None]

    def _directional_bias(self, 
                          group1_embedding, 
                          group2_embedding, 
                          comparison_embedding):
        # average embeddings for a group
        group_embeddings = [embedding.mean(0)[None,:] 
                    for embedding in [group1_embedding, group2_embedding]]
        # similarities
        comparison_relations = [cosine_similarity(embedding, comparison_embedding)
                            for embedding in group_embeddings]
        return (comparison_relations[0] - comparison_relations[1]).mean()

    def _normalized_directional_bias(self, 
                                     group1_embedding, 
                                     group2_embedding, 
                                     comparison_embedding):
        max_bias = self._directional_bias(
            group1_embedding, group2_embedding, group1_embedding)
        min_bias = self._directional_bias(
            group1_embedding, group2_embedding, group2_embedding)
        assert max_bias>0 and min_bias<0
        bias = self._directional_bias(
            group1_embedding, group2_embedding, comparison_embedding)
        normalized_bias = ((bias-min_bias)/(max_bias-min_bias)*2)-1
        return normalized_bias
