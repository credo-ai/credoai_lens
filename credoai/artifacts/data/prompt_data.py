import os

import pandas as pd

from credoai.utils.common import ValidationError
from credoai.utils.genAI_utils.prompt_utils import ANTHROPIC_PROMPTS, PROMPTS_PATHS

from .base_data import Data


class PromptData(Data):
    """Class wrapper for text prompts

    PromptData serves as an adapter between text prompts and the text prompt evaluator in Lens.

    Parameters
    -------------
    prompts : str
        Reference to a prompt dataset.
        Choices are:
        * builtin datasets, which include:
            'bold_gender', 'bold_political_ideology', 'bold_profession',
            'bold_race', 'bold_religious_ideology' (Dhamala et al. 2021)
            'realtoxicityprompts_1000', 'realtoxicityprompts_challenging_20',
            'realtoxicityprompts_challenging_100', 'realtoxicityprompts_challenging' (Gehman et al. 2020) and more!

            Find them all by calling :func:`~credoai.utils.get_builtin_prompts`
        * path of your own prompts csv file
         with the column 'prompt' and optionally column 'group'
         'Group' will be used for fairness assessments (disaggregating evaluations based on that group)
    name : str, optional
        Name of the prompts dataset. If None, the prompts value is used.
    size : int, optional
        Number of prompts to use. If None, all prompts are used.
    """

    def __init__(self, prompts: str, name: str = "", size: int = 0):
        if not name:
            name = prompts
        super().__init__("Prompt", name)
        self.prompts = prompts
        self.prompt_df = None
        self.size = size
        self._setup_prompt_data()

    def _setup_prompt_data(self):
        if self.prompts in ANTHROPIC_PROMPTS:
            df = pd.read_json(ANTHROPIC_PROMPTS[self.prompts], lines=True)
            df.rename(columns={"question": "prompt"}, inplace=True)
            self.prompt_df = df
            if self.size:
                df = df.sample(self.size)
            self.X = (df["prompt"] + "\nAnswer only Yes or No.").to_frame()
            self.y = df["answer_matching_behavior"].str.lstrip()
        else:
            if self.prompts in PROMPTS_PATHS:
                df = pd.read_csv(PROMPTS_PATHS[self.prompts])

            elif self.prompts.split(".")[-1] == "csv":
                df = pd.read_csv(self.prompts)
                cols_required = ["group", "subgroup", "prompt"]
                cols_given = list(df.columns)
                if set(cols_given) != set(cols_required):
                    cols_required_str = ", ".join(cols_required)
                    raise ValidationError(
                        "The provided prompts dataset csv file is not a valid file. Ensure it has all and only the following columns: "
                        + cols_required_str
                    )

            else:
                builtin_prompts_names = list(PROMPTS_PATHS.keys())
                builtin_prompts_names = ", ".join(builtin_prompts_names)
                raise Exception(
                    "The prompts dataset cannot be loaded. Ensure the provided prompts value is either a path to a valid csv file"
                    + " or name of one of the builtin datasets (i.e."
                    + builtin_prompts_names
                    + "). You provided "
                    + self.prompts
                )

            df.reset_index(drop=True, inplace=True)
            self.prompt_df = df
            if self.size:
                df = df.sample(self.size)
            if "group" in df:
                self.sensitive_features = df.pop("group")
            self.X = df
