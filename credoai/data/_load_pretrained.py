import joblib
import tensorflow_hub as hub
from credoai.data.utils import get_data_path

def load_lr_toxicity(verbose=True):
    lr_model_path = get_data_path('nlp_generator_analyzer/persisted_models/lr_toxicity.joblib')
    lr_model = joblib.load(lr_model_path)
    if verbose:
        print('Pretrained toxicity assessment model loaded.')
        
    use_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    if verbose:
        print('Universal Sentence Encoder loaded.')

    return {'model': lr_model, 'encoder': use_encoder}