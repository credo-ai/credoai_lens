"""Requires installation of requirements-extras.txt"""

from credoai.data import load_lr_toxicity
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
)

GENERATOR_CACHE = {}


def text_generator(prompt, tokenizer, model, num_sequences=1, model_kwargs=None):
    """Generates text for a given prompt, tokenizer, and model

    Parameters
    ----------
    prompt : str
        prompt text
    tokenizer : transformers Tokenizer
        a transformers tokenizer
    model : transformers Model
        a transformers model
    num_sequences : int
        the number of sequences to return for the input
    generate_kwargs : dict
        key words passed to a the text generator model

    Returns
    -------
    str
        response text
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    # generate responses
    kwargs = {
        "max_length": max(30, len(inputs[0]) + 1),
        "do_sample": True,
        "top_p": 0.95,
    }
    if model_kwargs is not None:
        kwargs.update(model_kwargs)
    kwargs["num_return_sequences"] = num_sequences
    outputs = model.generate(inputs, **kwargs)
    responses = [
        tokenizer.decode(o, skip_special_tokens=True)[len(prompt) :] for o in outputs
    ]
    return responses


def gpt1_text_generator(prompt, num_sequences=1, model_kwargs=None):
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
    tokenizer_gpt1 = GENERATOR_CACHE.get("tokenizer_gpt1")
    if tokenizer_gpt1 is None:
        tokenizer_gpt1 = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
        GENERATOR_CACHE["tokenizer_gpt1"] = tokenizer_gpt1
    # get or cache model
    model_gpt1 = GENERATOR_CACHE.get("model_gpt1")
    if model_gpt1 is None:
        model_gpt1 = OpenAIGPTLMHeadModel.from_pretrained(
            "openai-gpt", pad_token_id=tokenizer_gpt1.eos_token_id
        )
        GENERATOR_CACHE["model_gpt1"] = model_gpt1
    responses = text_generator(
        prompt,
        tokenizer_gpt1,
        model_gpt1,
        num_sequences=num_sequences,
        model_kwargs=model_kwargs,
    )
    return responses


def gpt2_text_generator(prompt, num_sequences=1, model_kwargs=None):
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
    tokenizer_gpt2 = GENERATOR_CACHE.get("tokenizer_gpt2")
    if tokenizer_gpt2 is None:
        tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
        GENERATOR_CACHE["tokenizer_gpt2"] = tokenizer_gpt2
    # get or cache model
    model_gpt2 = GENERATOR_CACHE.get("model_gpt2")
    if model_gpt2 is None:
        model_gpt2 = GPT2LMHeadModel.from_pretrained(
            "gpt2", pad_token_id=tokenizer_gpt2.eos_token_id
        )
        GENERATOR_CACHE["model_gpt2"] = model_gpt2
    responses = text_generator(
        prompt,
        tokenizer_gpt2,
        model_gpt2,
        num_sequences=num_sequences,
        model_kwargs=model_kwargs,
    )
    return responses


def get_demo_nlp_assessments():
    """Loads the default nlp assessment model
    The model assesses a text for toxicity

    Returns
    -------
    dict
        prediction model (sklearn.linear_model.LogisticRegression) and sentence encoding model (SentenceTransformer)
    """
    loaded = load_lr_toxicity()
    lr_model = loaded["model"]
    st_encoder = loaded["encoder"]

    def lr_assessment_fun(txt):
        txt_embedding = st_encoder.encode([txt])
        ypred = lr_model.predict_proba(txt_embedding)
        score = ypred[0][1]
        return score

    def verbosity_fun(txt):
        return len(txt) / 100

    return {"toxicity": lr_assessment_fun, "verbosity": verbosity_fun}
