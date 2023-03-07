import os
from typing import Union

import openai


class OpenAIAdapter:
    """
    This class is a wrapper around the OpenAI API for use with the :class:`GenText` model artifact.
    """

    def __init__(self, api_key=None, model="text-curie-001", openai_kwargs=None):
        """
        Parameters
        ----------
        api_key : str, optional
            OpenAI API key, by default None. If None assumes your api Key can
            be found in the OPENAI_API_KEY environment variable
        model : str, optional
            OpenAI model to use, by default "text-curie-001"
        openai_kwargs : dict, optional
            Additional kwargs to pass to the OpenAI API, by default None
        """

        if api_key is None:
            api_key = os.environ["OPENAI_API_KEY"]
        openai.api_key = api_key
        self.model = model
        self.kwargs = openai_kwargs or {"max_tokens": 256, "temperature": 0.4}

    def generate(self, prompt: Union[str, list]):
        """Returns a list of completions given a prompt"""
        resp = openai.Completion.create(model=self.model, prompt=prompt, **self.kwargs)
        return [r["text"].lstrip() for r in resp["choices"]]
