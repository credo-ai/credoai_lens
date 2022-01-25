import joblib
from credoai.data.utils import get_data_path
from sentence_transformers import SentenceTransformer

def load_lr_toxicity(verbose=True):
    lr_model_path = get_data_path('nlp_generator_analyzer/persisted_models/lr_toxicity.joblib')
    lr_model = joblib.load(lr_model_path)
    if verbose:
        print('Pretrained toxicity assessment model loaded.')

    st_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    if verbose:
        print('Sentence Encoder loaded.')

    return {'model': lr_model, 'encoder': st_encoder}