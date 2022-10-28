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
        self._model = None
        self._plan: Optional[dict] = None
        self._unique_tags: List[dict] = []

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

            # Extract unique tags
            for x in self._evidence_requirements:
                if x.tags not in self._unique_tags:
                    self._unique_tags.append(x.tags)
            self._unique_tags = [x for x in self._unique_tags if x]

            global_logger.info(
                f"Successfully registered with {len(self._evidence_requirements)} evidence requirements"
            )

            if self._unique_tags:
                global_logger.info(
                    f"The following tags have being found in the evidence requirements: {self._unique_tags}"
                )

            self.clear_evidence()

    def __parse_json_api(self, json_str):
        return deserialize(json.loads(json_str))

    @property
    def registered(self):
        return bool(self._plan)

    def add_evidence(self, evidences: Union[Evidence, List[Evidence]]):
        """
        Add evidences
        """
        self._evidences += wrap_list(evidences)

    def clear_evidence(self):
        self.set_evidence([])

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
        to_return = self._match_requirements()

        if filename is None:
            self._api_export()
        else:
            self._file_export(filename)

        if to_return:
            export_status = "All requirements were matched."
        else:
            export_status = "Partial match of requirements."

        global_logger.info(export_status)
        return to_return

    def get_evidence_requirements(self, tags: dict = None):
        """
        Returns evidence requirements. Each evidence requirement can have optional tags
        (a dictionary). Only returns requirements that have tags that match the model
        (if provided), which are tags with the same tags as the model, or no tags.

        Parameters
        ----------
        tags : dict, optional
            Tags to subset evidence requirements. If a model has been set, will default
            to the model's tags. Evidence requirements will be returned that have no
            tags or have the same tag as provided.

        Returns
        -------
        List[EvidenceRequirement]
        """
        if tags is None:
            tags = self.get_model_tags()

        reqs = [
            e for e in self._evidence_requirements if (not e.tags or e.tags == tags)
        ]
        return reqs

    def get_evidence_tags(self):
        """Return the unique tags used for all evidence requirements"""
        return self._unique_tags

    def get_model_tags(self):
        """Get the tags for the associated model"""
        if self._model:
            return self._model["tags"]
        else:
            return None

    def set_artifacts(self, model, training_dataset=None, assessment_dataset=None):
        """Sets up internal knowledge of model and datasets to send to Credo AI Platform"""
        global_logger.info(
            f"Adding model ({model.name}) to governance. Model has tags: {model.tags}"
        )
        prepared_model = {"name": model.name, "tags": model.tags}
        if training_dataset:
            prepared_model["training_dataset_name"] = training_dataset.name
        if assessment_dataset:
            prepared_model["assessment_dataset_name"] = assessment_dataset.name
        self._model = prepared_model

    def set_evidence(self, evidences: List[Evidence]):
        """
        Update evidences
        """
        self._evidences = evidences

    def tag_model(self, model):
        """Interactive utility to tag a model tags from assessment plan"""
        tags = self.get_evidence_tags()
        print(f"Select tag from assessment plan to associated with model:")
        print("0: No tags")
        for number, tag in enumerate(tags):
            print(f"{number+1}: {tag}")
        selection = int(input("Number of tag to associate: "))
        if selection == 0:
            selected_tag = None
        else:
            selected_tag = tags[selection - 1]
        print(f"Selected tag = {selected_tag}. Applying to model...")
        model.tags = selected_tag
        if self._model:
            self._model["tags"] = selected_tag

    def _api_export(self):
        global_logger.info(
            f"Uploading {len(self._evidences)} evidences.. for use_case_id={self._use_case_id} policy_pack_id={self._policy_pack_id}"
        )
        self._api.create_assessment(self._use_case_id, self._prepare_export_data())

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

    def _file_export(self, filename):
        global_logger.info(
            f"Saving {len(self._evidences)} evidences to {filename}.. for use_case_id={self._use_case_id} policy_pack_id={self._policy_pack_id} "
        )
        data = self._prepare_export_data()
        meta = {"client": "Credo AI Lens", "version": __version__}
        data = json_dumps(serialize(data=data, meta=meta))
        with open(filename, "w") as f:
            f.write(data)

    def _match_requirements(self):
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

    def _prepare_export_data(self):
        evidences = self._prepare_evidences()
        data = {
            "policy_pack_id": self._policy_pack_id,
            "models": [self._model] if self._model else None,
            "evidences": evidences,
            "$type": "assessments",
        }
        return data

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
