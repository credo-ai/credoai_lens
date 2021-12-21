import json
import numpy as np
import pandas as pd
from absl import logging
from credoai.modules.credo_module import CredoModule
from credoai.utils.common import wrap_list
from credoai.data.utils import get_data_path
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
        return biases

    def prepare_results(self, group1='male', group2='female'):
        return self.run(group1, group2)
    
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


class NLPGeneratorAnalyzer(CredoModule):
    """
    NLP generation analyzer for Credo AI

    This module takes in a function that generates and returns a text response
    for a text prompt. Using this function, it calculates
    toxicity and bias metrics similar to Dhamala et al. (2021).

    Parameters
    ----------
    generation_fun : callable
        function that takes in a string (prompt) and outputs a string (response)
    assessment_fun : callable
        function that assesses a string (prompt) and outputs its score (float between 0 and 1)
    assessment_attribute : str
        the attribute of text that is assessed (e.g., toxicity)
    """
    def __init__(self,
            generation_fun,
            assessment_fun,
            assessment_attribute
            ):
        self.generation_fun = generation_fun
        self.assessment_fun = assessment_fun
        self.assessment_attribute = assessment_attribute
        self.raw_results = None
        self.religious_prompts = None

    def prepare_results(self):
        """Generates summary statistics of raw assessment results generated by self.run

        Returns
        -------
        pandas.dataframe
            Summary statistics of assessment results

        Raises
        ------
        Exception
            Occurs is self.run is not called yet to generate the raw assessment results
        """        
        if self.raw_results is not None:
            # Calculate statistics across groups
            results = self.raw_results[['group',self.assessment_attribute]].groupby('group', as_index=False).mean()
            results.loc[len(results)] = ['overall', self.raw_results[self.assessment_attribute].mean()] 
            results['protected_attribute'] = self.raw_results.iloc[0]['protected_attribute']
            results.rename(columns={self.assessment_attribute:'value'}, inplace=True)
            results['assessment_attribute'] = self.assessment_attribute
            results = results[['assessment_attribute', 'group', 'protected_attribute', 'value']]
            return results
        else:
            raise Exception(
                "This NLPGeneratorAnalyzer instance is not run yet. Call 'run' with appropriate arguments before using this module."
            )

    def run(self, protected_attribute='religion', n_iterations=1, groups='all'):
        """Analyze a text generation model for assessment attribute across groups of a protected attribute

        Parameters
        ----------
        protected_attribute : str
            protected attribute for which the analysis should be done
            possible values: 'religion'
        n_iterations : int
            number of times to generate responses for a prompt
            set to 1 if the generation model is not stochastic
        groups : list or 'all'
            protected attribute groups to include in the analysis
            Possible values:
            protected_attribute='religion': ['judaism', 'christianity', 'islam', 'hinduism', 'buddhism', 'sikhism', 'atheism']

        Returns
        -------
        pandas.dataframe
            Assessment attribute values for all the prompts and iterations
        """        
        if protected_attribute == 'religion':
            res = self._analyze_religion(n_iterations, groups)
            res['protected_attribute'] = 'religion'

        self.raw_results = res

        return self

    def _generation_fun_robust(self, prompt):
        """Makes process robust to when generation_fun fails for a prompt
        Failed cases are tracked and excluded from the follow-up assessment

        Parameters
        ----------
        prompt : str
            prompt text

        Returns
        -------
        str
            response text
        """        
        try:
            return self.generation_fun(prompt)
        except:
            return 'e'


    def _analyze_religion(self, n_iterations, groups):
        """Analyze a text generation model for assessment attribute across religion groups

        Parameters
        ----------
        n_iterations : int
            number of times to generate responses for a prompt
            set to 1 if the generation model is not stochastic
        groups : list or 'all'
            religion groups to include in the analysis
            Possible values: ['judaism', 'christianity', 'islam', 'hinduism', 'buddhism', 'sikhism', 'atheism']

        Returns
        -------
        pandas.dataframe
            assessment attribute scores for all the religion prompts and iterations
        """
        # read prompts
        if self.religious_prompts is None:
            religious_prompts_path = get_data_path('nlp_generator_analyzer/prompts/religious_ideology_prompt.json')
            with open(religious_prompts_path, "r") as read_file:
                self.religious_prompts = json.load(read_file)

        # convert the prompts json to a dataframe for better handling
        df = pd.DataFrame(columns=['group', 'subgroup', 'prompt'])
        for k,v in self.religious_prompts.items():
            for k2,v2 in v.items():
                temp = pd.DataFrame.from_dict(v2)
                temp.rename(columns={0: "prompt" }, inplace=True)
                temp['group'] = k
                temp['subgroup'] = k2
                df = df.append(temp)

        # Generate and record responses for the prompts n_iterations times
        # Each run may take several minutes to complete
        dfruns = pd.DataFrame(columns=['group', 'subgroup', 'prompt', 'run'])
        if groups != 'all':
            df = df[df['group'].isin(groups)]
        
        for i in range(n_iterations):
            logging.info('Performing Iteration ' + str(i+1) + ' of ' + str(n_iterations) + ' for religion') 
            df['response'] = df['prompt'].apply(self._generation_fun_robust)
            df['run'] = i+1
            dfruns = dfruns.append(df)

        # Assess the responses
        dfrunst = dfruns[dfruns['response'] != 'e'].copy()  # exclude cases where generator failed to generate a response
        dfrunst[self.assessment_attribute] = dfrunst['response'].apply(self.assessment_fun)
    
        return dfrunst
            



        
