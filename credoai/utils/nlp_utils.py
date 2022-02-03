"""Requires installation of requirements-extras.txt"""

from transformers import (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, 
                          GPT2LMHeadModel, GPT2Tokenizer)
from credoai.data import load_lr_toxicity

def text_generator(prompt, tokenizer, model):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max(30, len(inputs[0])+1), do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
    
def gpt1_text_generator(prompt):
    tokenizer_gpt1 = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    model_gpt1 = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt', pad_token_id=tokenizer_gpt1.eos_token_id)
    text_generator(prompt, tokenizer_gpt1, model_gpt1)
    return response

def gpt2_text_generator(prompt):
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
    model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', 
                    pad_token_id=GPT2_TOKENIZER.eos_token_id)
    text_generator(prompt, tokenizer_gpt2, model_gpt2)
    return response

def get_default_nlp_assessments():
    loaded = load_lr_toxicity()
    lr_model = loaded['model']
    st_encoder = loaded['encoder']
    def lr_assessment_fun(txt):
        txt_embedding = st_encoder.encode([txt])
        ypred = lr_model.predict_proba(txt_embedding)
        score = ypred[0][1]
        return score
    return {'toxicity': lr_assessment_fun}