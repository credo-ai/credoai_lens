"""Requires installation of requirements-extras.txt"""

from transformers import (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, 
                          GPT2LMHeadModel, GPT2Tokenizer)
from credoai.data import load_lr_toxicity

GENERATOR_CACHE = {}

def text_generator(prompt, tokenizer, model):
    """Generates text for a given prompt, tokenizer, and model

    Parameters
    ----------
    prompt : str
        prompt text
    tokenizer : transformers Tokenizer
        a transformers tokenizer
    model : transformers Model
        a transformers model     
    
    Returns
    -------
    str
        response text
    """  
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max(30, len(inputs[0])+1), do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
    return response
    
def gpt1_text_generator(prompt):
    """Generates text for a given prompts using GPT1 model

    Parameters
    ----------
    prompt : str
        prompt text

    Returns
    -------
    str
        response text
    """ 
    # get or cache tokenizer
    tokenizer_gpt1 = GENERATOR_CACHE.get('tokenizer_gpt1')
    if tokenizer_gpt1 is None:
        tokenizer_gpt1 = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        GENERATOR_CACHE['tokenizer_gpt1'] = tokenizer_gpt1
    # get or cache model
    model_gpt1 = GENERATOR_CACHE.get('model_gpt1')
    if model_gpt1 is None:
        model_gpt1 = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt', pad_token_id=tokenizer_gpt1.eos_token_id)
        GENERATOR_CACHE['model_gpt1'] = model_gpt1
    response = text_generator(prompt, tokenizer_gpt1, model_gpt1)
    return response

def gpt2_text_generator(prompt):
    """Generates text for a given prompts using GPT2 model
    
    Parameters
    ----------
    prompt : str
        prompt text
    
    Returns
    -------
    str
        response text
    """   
        # get or cache tokenizer
    tokenizer_gpt2 = GENERATOR_CACHE.get('tokenizer_gpt2')
    if tokenizer_gpt2 is None:
        tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
        GENERATOR_CACHE['tokenizer_gpt2'] = tokenizer_gpt2
    # get or cache model
    model_gpt2 = GENERATOR_CACHE.get('model_gpt2')
    if model_gpt2 is None:
        model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer_gpt2.eos_token_id)
        GENERATOR_CACHE['model_gpt2'] = model_gpt2
    response = text_generator(prompt, tokenizer_gpt2, model_gpt2)
    return response

def get_demo_nlp_assessments():
    """Loads the default nlp assessment model
    The model assesses a text for toxicity

    Returns
    -------
    dict
        prediction model (sklearn.linear_model.LogisticRegression) and sentence encoding model (SentenceTransformer)
    """   
    loaded = load_lr_toxicity()
    lr_model = loaded['model']
    st_encoder = loaded['encoder']
    def lr_assessment_fun(txt):
        txt_embedding = st_encoder.encode([txt])
        ypred = lr_model.predict_proba(txt_embedding)
        score = ypred[0][1]
        return score
    def verbosity_fun(txt):
        return len(txt)/100
    return {'toxicity': lr_assessment_fun,
            'verbosity': verbosity_fun}
