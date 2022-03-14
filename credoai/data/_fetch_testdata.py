import numpy as np
import pandas as pd

def fetch_testdata(add_nan=False):
    """Returns testing data for Lens"""
    df = pd.DataFrame(
        {
            "gender": ['f', 'f', 'f', 'f', 'f', 'f', 'm', 'm', 'm', 'm', 'm', 'm'],
            "experience": [0, 0.1, 0.2, 0.4, 0.5, 0.6, 0, 0.1, 0.2, 0.4, 0.5, 0.6],
        }
    )
    if add_nan:
        df = df.mask(np.random.random(df.shape) < .1)

    target = pd.Series([0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1], name='income')
    data = {
        'data': df,
        'target': target
    }
    return data