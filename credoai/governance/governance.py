"""
Credo Governance
"""

import json
from typing import List, Optional, Union

from credoai import __version__
from credoai.evidence import Evidence, EvidenceRequirement
from credoai.utils import (
    ValidationError,
    check_subset,
    global_logger,
    json_dumps,
    wrap_list,
)
from json_api_doc import deserialize, serialize

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

    Examples
    --------
    If you want to use your own configuration:

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

    """

    def __init__(
        self, config_path: str = None, credo_api_client: CredoApiClient = None
    ):
        """Governance object to connect Lens with Credo AI Platform

        Parameters
        ----------
        config_path : str, optional
            path to .credoconfig file. If None, points to ~/.credoconfig, by default None
        credo_api_client : CredoApiClient, optional
            If provided, overrides the API configuration defined by
            the config path, by default None
        """
        self._use_case_id: Optional[str] = None
        self._policy_pack_id: Optional[str] = None
        self._evidence_requirements: List[EvidenceRequirement] = []
        self._evidences: List[Evidence] = []
        self._plan: Optional[dict] = None

        if credo_api_client:
            client = credo_api_client
        else:
            client = CredoApiClient(config_path=config_path)

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

        Examples
        --------
        Get assessment plan and register it.
        There are three ways to do it:

            1. With assessment_plan_url.

                gov.register(assessment_plan_url="https://api.credo.ai/api/v2/tenant/use_cases/{id}/assessment_plans/{pp_id}")

            2. With use case name and policy pack key.

                gov.register(use_case_name="Fraud Detection", policy_pack_key="FAIR")

            3. With assessment_plan json string or filename. It is used in the air-gap condition.

                gov.register(assessment_plan="JSON_STRING")
                gov.register(assessment_plan_file="FILENAME")

        After successful registration, `gov.registered` returns True and able to get evidence_requirements:

            gov.registered    # returns True
            gov.get_evidence_requirements()


        """
        self._plan = None

        plan = None
        if use_case_name:
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

            self.clear_evidence()

    def __parse_json_api(self, json_str):
        return deserialize(json.loads(json_str))

    @property
    def registered(self):
        return bool(self._plan)

    def get_evidence_requirements(self):
        """
        Returns evidence requirements


        Returns
        -------
        List[EvidenceRequirement]
        """
        return self._evidence_requirements

    def clear_evidence(self):
        self.set_evidence([])

    def set_evidence(self, evidences: List[Evidence]):
        """
        Update evidences
        """
        self._evidences = evidences

    def add_evidence(self, evidences: Union[Evidence, List[Evidence]]):
        """
        Add evidences
        """
        self._evidences += wrap_list(evidences)

    def match_requirements(self):
        missing = []
        required_labels = [e.label for e in self.get_evidence_requirements()]
        for label in required_labels:
            matching_evidence = self._check_inclusion(label, self._evidences)
            if not matching_evidence:
                missing.append(label)
                global_logger.info(f"Missing required evidence with label ({label}).")
            else:
                matching_evidence[0].label = label
        return not bool(missing)

    def export(self, filename=None):
        """
        Upload evidences to CredoAI Governance(Report) App

        Returns
        -------
        True
            When uploading is successful with all evidence
        False
            When it is not registered yet, or evidence is insufficient
        """
        if not self._validate_export():
            return False
        to_return = self.match_requirements()

        evidences = self._prepare_evidences()

        if filename is None:
            self._api_export(evidences)
        else:
            self._file_export(evidences, filename)

        if to_return:
            export_status = "All requirements were matched."
        else:
            export_status = "Partial match of requirements."

        global_logger.info(export_status)
        return to_return

    def _api_export(self, evidences):
        global_logger.info(
            f"Uploading {len(evidences)} evidences.. for use_case_id={self._use_case_id} policy_pack_id={self._policy_pack_id}"
        )
        self._api.create_assessment(self._use_case_id, self._policy_pack_id, evidences)

    def _check_inclusion(self, label, evidence):
        matching_evidence = []
        for e in evidence:
            if check_subset(label, e.label):
                matching_evidence.append(e)
        if not matching_evidence:
            return False
        if len(matching_evidence) > 1:
            global_logger.error(
                "Multiple evidence labels were found matching one requirement"
            )
            return False
        return matching_evidence

    def _file_export(self, evidences, filename):
        global_logger.info(
            f"Saving {len(evidences)} evidences to {filename}.. for use_case_id={self._use_case_id} policy_pack_id={self._policy_pack_id} "
        )
        data = {
            "policy_pack_id": self._policy_pack_id,
            "evidences": evidences,
            "$type": "assessments",
        }
        meta = {"client": "Credo AI Lens", "version": __version__}
        data = json_dumps(serialize(data=data, meta=meta))
        with open(filename, "w") as f:
            f.write(data)

    def _prepare_evidences(self):
        evidences = list(map(lambda e: e.struct(), self._evidences))
        return evidences

    def _validate_export(self):
        if not self.registered:
            global_logger.info("Governance is not registered, please register first")
            return False

        if 0 == len(self._evidences):
            global_logger.info(
                "No evidences added to governance, please add evidences first"
            )
            return False
        return True
