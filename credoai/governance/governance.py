"""
Credo Governance
"""

import json

from credoai.evidence.evidence import Evidence
from credoai.evidence.evidence_requirement import EvidenceRequirement
from credoai.utils import global_logger
from json_api_doc import deserialize

from .credo_api import CredoApi
from .credo_api_client import CredoApiClient


class Governance:
    """Class to store governance data.

    Governance is used to interact with the CredoAI Governance(Report) App.
    It has two main jobs.
    1. Get evidence_requirements of use_case and policy pack.
    2. Upload evidences gathered with evidence_requirements

    Parameters
    ----------
    credo_api_client: CredoApiClient, optional
        Credo API client. Uses default Credo API client if it is None
        Default Credo API client uses `~/.credo_config` to read API server configuration.
        Please prepare `~/.credo_config` file by downloading it from CredoAI Governance App.(My Settings > Tokens)

        If you want to use your own configuration,

        ```python
        from credoai.governance.credo_api_client import CredoApiClient, CredoApiConfig
        from credoai.governance.goverance import Governance

        config = CredoApiConfig(
            api_key="API_KEY", tenant="credo", api_server="https://api.credo.ai"
        )
        # or using credo_config file
        config = CredoApiConfig()
        config.load_config("CREDO_CONFIG_FILE_PATH")

        client = CredoApiClient(config=config)
        governace = Governance(credo_api_client=client)
        ```

    """

    def __init__(self, credo_api_client: CredoApiClient = None):
        self._use_case_id: str = None
        self._policy_pack_id: str = None
        self._evidence_requirements: list[EvidenceRequirement] = []
        self._evidences: list[Evidence] = []
        self._plan: dict = None

        if credo_api_client:
            client = credo_api_client
        else:
            client = CredoApiClient()

        self._api = CredoApi(client=client)

    def register(
        self,
        assessment_plan_url: str = None,
        use_case_name: str = None,
        policy_pack_key: str = None,
        assessment_plan: str = None,
        assessment_plan_file: str = None,
    ):
        """
        Get assessment plan and register it.
        There are three ways to do it
        1. With assessment_plan_url.
        ```
        gov.register(assessment_plan_url="https://api.credo.ai/api/v2/tenant/use_cases/{id}/assessment_plans/{pp_id}")
        ```
        2. With use case name and policy pack key.
        ```
        gov.register(use_case_name="Fraud Detection", policy_pack_key="FAIR")
        ```
        3. With assessment_plan json string or filename. It is used in the air-gap condition.
        ```
        gov.register(assessment_plan="JSON_STRING")
        gov.register(assessment_plan_file="FILENAME")
        ```

        Afeter successful registration, `gov.registered` returns True and able to get evidence_requirements
        ```
        gov.registered    # returns True
        gov.get_evidence_requirements()
        ```

        Parameters
        ----------
        assessment_plan_url : str
            assessment plan URL
        use_case_name : str
            use case name
        policy_pack_key : str
            policy pack key
        assessment_plan : str
            assessment plan JSON string
        assessment_plan_file : str
            assessment plan file name that holds assessment plan JSON string
        """
        self._plan = None

        plan = None
        if use_case_name and policy_pack_key:
            assessment_plan_url = self._api.get_assessment_plan_url(
                use_case_name, policy_pack_key
            )

        if assessment_plan_url:
            plan = self._api.get_assessment_plan(assessment_plan_url)

        if assessment_plan:
            plan = self.__parse_json_api(assessment_plan)

        if assessment_plan_file:
            with open(assessment_plan_file, "r") as f:
                json_str = f.read()
                plan = self.__parse_json_api(json_str)

        if plan:
            self._plan = plan
            self._use_case_id = plan.get("use_case_id")
            self._policy_pack_id = plan.get("policy_pack_id")
            self._evidence_requirements = list(
                map(
                    lambda d: EvidenceRequirement(d),
                    plan.get("evidence_requirements", []),
                )
            )

            global_logger.info(
                f"Successfully registered with {len(self._evidence_requirements)} evidence requirements"
            )

            self.set_evidences([])

    def __parse_json_api(self, json_str):
        return json.loads(json_str)

    @property
    def registered(self):
        return bool(self._plan)

    def get_evidence_requirements(self):
        """
        Returns evidence requirements


        Returns
        -------
        list[EvidenceRequirement]
        """
        return self._evidence_requirements

    def set_evidences(self, evidences: list[Evidence]):
        """
        Update evidences
        """
        self._evidences = evidences

    def add_evidences(self, evidences: list[Evidence]):
        """
        Add evidences
        """
        self._evidences = self._evidences + evidences

    def export(self):
        """
        Upload evidences to CredoAI Governance(Report) App

        Returns
        -------
        True
            When uploading is successful
        False
            When it is not registered yet, or there is no evidence
        """

        if not self.registered:
            global_logger.info("It is not registered, please register first")
            return False

        if 0 == len(self._evidences):
            global_logger.info("No evidences found, please add evidences first")
            return False

        evidences = list(map(lambda e: e.struct(), self._evidences))

        global_logger.info(
            f"Uploading {len(evidences)} evidences.. for use_case_id={self._use_case_id} policy_pack_id={self._policy_pack_id}"
        )
        self._api.create_assessment(self._use_case_id, self._policy_pack_id, evidences)

        return True
