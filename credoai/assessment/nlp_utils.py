from transformers import GPT2LMHeadModel, GPT2Tokenizer
from credoai.data import load_lr_toxicity

GPT2_TOKENIZER = GPT2Tokenizer.from_pretrained('gpt2')
GPT2_MODEL = GPT2LMHeadModel.from_pretrained('gpt2', 
                pad_token_id=GPT2_TOKENIZER.eos_token_id)

def gpt2_text_generator(prompt):
    inputs = GPT2_TOKENIZER.encode(prompt, return_tensors='pt')
    outputs = GPT2_MODEL.generate(inputs, max_length=max(30, len(inputs[0])+1), do_sample=True)
    response = GPT2_TOKENIZER.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
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