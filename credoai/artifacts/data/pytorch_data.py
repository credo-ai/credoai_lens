from typing import Union

import numpy as np

try:
    import torch
except ImportError:
    print(
        "Torch not loaded. Torch models will not be wrapped properly if supplied to ClassificationModel"
    )

from credoai.utils.common import ValidationError

from .base_data import Data


class PytorchData(Data):
    """Class wrapper around Pytorch data

    Pytorch serves as an adapter between pytorch dataloaders
    and the evaluators in Lens.

    Lens dataset evaluator functionality (e.g. DataFairness, DataProfiling) is dependent on
    loading your data into memory, which this class does not do. If you want to use those
    evaluators consider passing your entire dataset (or batches) to the relevant evaluators.

    Parameters
    -------------
    name : str
        Label of the dataset
    dataloader : torch.utils.data.DataLoader, optional
        DataLoader object for PyTorch models, which provides an iterable over the given dataset.
        If provided, DataLoader will be used to extract labels and features will be passed to the model
        for predictions during the assessment. This option is specifically designed for compatibility
        with PyTorch models, and it doesn't support data analysis like data profiling.
    sensitive_features : pd.Series, pd.DataFrame, optional
        Sensitive Features, which will be used for disaggregating performance
        metrics. This can be the feature you want to perform segmentation analysis on, or
        a feature related to fairness like 'race' or 'gender'. Sensitive Features *must*
        be categorical features.
    sensitive_intersections : bool, list
        Whether to add intersections of sensitive features. If True, add all possible
        intersections. If list, only create intersections from specified sensitive features.
        If False, no intersections will be created. Defaults False
    """

    def __init__(
        self,
        name: str,
        dataloader=None,
        sensitive_features=None,
        sensitive_intersections: Union[bool, list] = False,
    ):
        self.dataloader = dataloader
        X, y = self._process_dataloader(dataloader)
        self._validate_dataloader()
        super().__init__(
            "PyTorch",
            name=name,
            X=X,
            y=y,
            sensitive_features=sensitive_features,
            sensitive_intersections=sensitive_intersections,
        )

    def _process_dataloader(self, dataloader):
        labels_list = []
        for _, labels in self.dataloader:
            labels_list.append(labels.numpy())
        y = np.concatenate(labels_list, axis=0)
        # self.X will be sorted out in the predict functions
        # This does not support any sort of data analysis like profiling!!
        X = dataloader
        return X, y

    def _validate_dataloader(self):
        if not isinstance(self.dataloader, torch.utils.data.DataLoader):
            raise ValidationError(
                f"`dataloader` must be of type torch.utils.data.DataLoader. Supplied dataloader is of type {type(self.dataloader)}"
            )
