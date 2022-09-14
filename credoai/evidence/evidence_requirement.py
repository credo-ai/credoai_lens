from abc import ABC


class EvidenceRequirement(ABC):
    def __init__(
        self, data: dict,
    ):
        self._evidence_type: str = data.get("evidence_type")
        self._label: dict = data.get("label")
        self._sensitive_features: list[str] = data.get("sensitive_features", [])

    @property
    def evidence_type(self):
        return self._evidence_type

    @property
    def label(self):
        return self._label

    @property
    def sensitive_features(self):
        return self._sensitive_features

