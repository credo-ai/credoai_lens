import os
from glob import glob

from credoai.utils import get_project_root

PROMPTS_PATHS = {}
data_path = os.path.join(get_project_root(), "data")
for prompt_path in glob(os.path.join(data_path, "text_prompts/*csv")):
    prompt_name = os.path.basename(prompt_path)[:-4]
    PROMPTS_PATHS[prompt_name] = prompt_path

ANTHROPIC_PROMPTS = {}
anthropic_evals_path = os.path.join(get_project_root(), "utils/genAI_utils/evals")
for prompt_path in glob(os.path.join(anthropic_evals_path, "persona/*jsonl")):
    prompt_name = "anthropiceval_" + os.path.basename(prompt_path)[:-6]
    ANTHROPIC_PROMPTS[prompt_name] = prompt_path


def get_builtin_prompts():
    """Get a list of all builtin prompts datasets"""
    all_prompts = list(PROMPTS_PATHS.keys()) + list(ANTHROPIC_PROMPTS.keys())
    return sorted(all_prompts)
