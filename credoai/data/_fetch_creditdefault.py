import pandas as pd
from sklearn.datasets import fetch_openml
from .utils import get_local_credo_path
from credoai.utils.common import SupressSettingWithCopyWarning

FEATURES = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
CATEGORICAL_FEATURES = ['EDUCATION', 'MARRIAGE','PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
TARGET = ['default payment next month']

def fetch_creditdefault(*, cache=True, data_home=None,
                as_frame=True, return_X_y=False):
    """Load the UCI Adult dataset (binary classification).
    Download it if necessary.
    ==============   ==============
    Samples total             48842
    Dimensionality               14
    Features                   real
    Classes                       2
    ==============   ==============
    Source: UCI Repository [1]_ , Paper: R. Kohavi (1996) [2]_
    Prediction task is to determine whether a person makes over $50,000 a
    year.
    .. versionadded:: 0.5.0
    Parameters
    ----------
    cache : bool, default=True
        Whether to cache downloaded datasets using joblib.
    data_home : str, default=None
        Specify another download and cache folder for the datasets.
        By default, all scikit-learn data is stored in '~/.fairlearn-data'
        subfolders.
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string or categorical). The target is
        a pandas DataFrame or Series depending on the number of target_columns.
        The Bunch will contain a ``frame`` attribute with the target and the
        data. If ``return_X_y`` is True, then ``(data, target)`` will be pandas
        DataFrames or Series as describe above.
    return_X_y : bool, default=False
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.
    Returns
    -------
    dataset : :obj:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        data : ndarray, shape (48842, 14)
            Each row corresponding to the 14 feature values in order.
            If ``as_frame`` is True, ``data`` is a pandas object.
        target : numpy array of shape (48842,)
            Each value represents whether the person earns more than $50,000
            a year (>50K) or not (<=50K).
            If ``as_frame`` is True, ``target`` is a pandas object.
        feature_names : list of length 14
            Array of ordered feature names used in the dataset.
        DESCR : string
            Description of the UCI Adult dataset.
    (data, target) : tuple of (numpy.ndarray, numpy.ndarray) or (pandas.DataFrame, pandas.Series)
        if ``return_X_y`` is True and ``as_frame`` is False
    (data, target) : tuple of (pandas.DataFrame, pandas.Series)
        if ``return_X_y`` is True and ``as_frame`` is True
    References
    ----------
    .. [1] Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques 
        for the predictive accuracy of probability of default of credit card clients. 
        Expert Systems with Applications, 36(2), 2473-2480.
        Available: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
    """
    if not data_home:
        data_home = get_local_credo_path()
    
    output = fetch_openml(
        data_id=42477,
        data_home=data_home,
        cache=cache,
        as_frame=as_frame,
        return_X_y=return_X_y,
    )
    
    output['feature_names'] = FEATURES
    output['target_names'] = TARGET
    if as_frame:
        dataset = output['data']
        dataset.columns = FEATURES
        with SupressSettingWithCopyWarning():
            dataset['SEX'] = dataset['SEX'].astype('category')\
                .cat.rename_categories(['female','male'])
        output['target'].name = TARGET[0]
        output['target'] = output['target'].cat.rename_categories([0,1])        
        with SupressSettingWithCopyWarning():
            for col in CATEGORICAL_FEATURES:
                dataset[col] = dataset[col].astype('category')
    return output

