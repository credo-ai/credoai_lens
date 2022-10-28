import pytest
from credoai.governance.credo_api import CredoApi
from credoai.governance.credo_api_client import CredoApiClient


class TestCredoApi:
    @pytest.fixture()
    def client(self, mocker):
        # Use mock for CredoApiClient becasue we do not want to call HTTP request from test.
        # We will only check if CredoApi is calling client method with right arguments.
        mocker.patch.object(CredoApiClient, "get")
        mocker.patch.object(CredoApiClient, "post")
        mocker.patch.object(CredoApiClient, "patch")
        mocker.patch.object(CredoApiClient, "delete")
        mocker.patch.object(CredoApiClient, "refresh_token")

        return CredoApiClient()

    @pytest.fixture()
    def api(self, client):
        return CredoApi(client=client)

    # def test_real(self):
    #     client = CredoApiClient()
    #     api = CredoApi(client=client)

    #     response = api.get_assessment_plan(
    #         "http://localhost:4000/api/v2/credoai/use_cases/64YUaLWSviHgibJaRWr3ZE/assessment_plans/NYCE+1"
    #     )
    #     print(response)

    #     response = api.get_assessment_plan_url("Keyboard Sensitivity", "NYCE")
    #     print(response)

    def test_get_assessment_plan_url(self, api, client):
        api.get_assessment_plan_url("Fraud Detection", "FAIR")

        client.get.assert_called_with(
            "assessment_plan_url?use_case_name=Fraud Detection&policy_pack_key=FAIR"
        )

    def test_get_assessment_plan(self, api, client):
        url = "http://localhost:4000/api/v2/credoai/use_cases/64YUaLWSviHgibJaRWr3ZE/assessment_plans/NYCE+1"
        api.get_assessment_plan(url)

        client.get.assert_called_with(url)

    def test_create_assessment(self, api, client):
        use_case_id = "64YUaLWSviHgibJaRWr3ZE"
        policy_pack_id = "FAIR+1"

        evidences = [
            {
                "type": "metric",
                "label": {"metric_type": "accuracy_score"},
                "metadata": {
                    "model_name": "MODEL NAME 1",
                    "dataset_name": "DATASET NAME 1",
                },
                "data": {
                    "value": 0.6,
                    "confidence_interval_range": [0.1, 0.5],
                    "confidence_interval": 0.2,
                },
                "generated_at": "2022-05-03T11:33:25.582138Z",
            }
        ]
        body = {
            "policy_pack_id": policy_pack_id,
            "evidences": evidences,
            "$type": "assessments",
        }
        api.create_assessment(use_case_id, body)

        client.post.assert_called_with(f"use_cases/{use_case_id}/assessments", body)
