"""
Credo API functions
"""
from requests.exceptions import HTTPError
from credoai.utils import global_logger
from credoai.governance.credo_api_client import CredoApiClient
from credoai.evidence.evidence import Evidence


class CredoApi:
    """
    CredoApi holds Credo API functions
    """

    def __init__(self, client: CredoApiClient = None):
        self._client = client

    def set_client(self, client: CredoApiClient):
        """
        Sets Credo Api Client

        Parameters
        ----------
        client : CredoApiClient
            Credo API client
        """
        self._client = client

    def get_assessment_plan_url(self, use_case_name: str, policy_pack_key: str):
        """
        Convert use_case_name and policy_pack_key to assessment_plan_url

        Parameters
        ----------
        use_case_name : str
            name of a use case
        policy_pack_key : str
            policy pack key, ie: FAIR

        Returns
        -------
        None
            When use_case_name does not exist or policy_pack_key is not registered to the use_case
        str
            assessment_plan_url

        Raises
        ------
        HTTPError
            When API request returns error other than 404
        """

        try:
            path = f"assessment_plan_url?use_case_name={use_case_name}&policy_pack_key={policy_pack_key}"
            response = self._client.get(path)
            return response["url"]
        except HTTPError as error:
            if error.response.status_code == 404:
                global_logger.info(
                    f"Use case ({use_case_name}) with policy pack({policy_pack_key}) does not exist"
                )
                return None
            raise error

    def get_assessment_plan(self, url: str):
        """
        Get assessment plan from API server and returns it.

        Parameters
        ----------
        url : str
            assessment plan URL

        Returns
        -------
        dict
            evidence_requirements(list): list of evidence requirements
            policy_pack_id(str): policy pack id(key+version), ie: FAIR+1
            use_case_id(str): use case id

        Raises
        ------
        HTTPError
            When API request returns error
        """

        return self._client.get(url)

    def create_assessment(
        self, use_case_id: str, policy_pack_id: str, evidences: list[dict]
    ):
        """
        Upload evidences to API server.

        Parameters
        ----------
        use_case_id : str
            use case id
        policy_pack_id : str
            policy pack id, ie: FAIR+1
        evidences: list[dict]
            list of evidences

        Raises
        ------
        HTTPError
            When API request returns error
        """

        path = f"use_cases/{use_case_id}/assessments"

        # list(map(lambda e: e.struct(), evidences))
        data = {
            "policy_pack_id": policy_pack_id,
            "evidences": evidences,
        }
        return self._client.post(path, data)
