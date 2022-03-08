"""
Modules are tools that can be used together to assess ML systems.

Modules are conditionally imported based on whether their required
packages are installed. requirements.txt covers a base set of modules,
but more requirements are needed for different modules. See README.md
"""

import importlib
import inspect
base = 'credoai.modules'
modules = [
    'dataset_modules.dataset_fairness',
    'model_modules.fairness_nlp',
    'model_modules.fairness_base',
    'model_modules.nlp_generator'
]

importable_modules = []
__all__ = []

def get_module_classes(module):
    classes = inspect.getmembers(module, lambda member: inspect.isclass(member) 
        and member.__module__ == module.__name__)
    return [c[0] for c in classes]

# try to import each module
# make the ones that are successful
# importable with "from credoai.module import {module}"
for module in modules:
    module_path = f'{base}.{module}'
    try:
        mod = importlib.import_module(module_path)
        importable_modules.append(module_path)\
        # emulate from mod import *
        module_classes = get_module_classes(mod)
        __all__ += module_classes
        globals().update({k: getattr(mod, k) for k in module_classes})
    except ModuleNotFoundError:
        pass

