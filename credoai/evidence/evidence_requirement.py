from abc import ABC
from typing import List


class EvidenceRequirement(ABC):
    def __init__(
        self,
        data: dict,
    ):
        self._evidence_type: str = data.get("evidence_type")
        self._label: dict = data.get("label")
        self._sensitive_features: List[str] = data.get("sensitive_features", [])
        self._tags: dict = data.get("tags")

    def __str__(self) -> str:
        return f"{self.evidence_type}-EvidenceRequirement.label-{self.label}"

    @property
    def evidence_type(self):
        return self._evidence_type

    @property
    def label(self):
        return self._label

    @property
    def tags(self):
        return self._tags
