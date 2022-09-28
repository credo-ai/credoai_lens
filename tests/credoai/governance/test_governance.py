import json
import tempfile

import pytest
from credoai.evidence.evidence import MetricEvidence, TableEvidence
from credoai.governance.credo_api import CredoApi
from credoai.governance.credo_api_client import CredoApiClient
from credoai.governance.governance import Governance
from pandas import DataFrame

USE_CASE_ID = "64YUaLWSviHgibJaRWr3ZE"
POLICY_PACK_ID = "NYCE+1"
EVIDENCE_REQUIREMENTS = [
    {"evidence_type": "metric", "label": {"metric_type": "accuracy_score"}},
    {"evidence_type": "metric", "label": {"metric_type": "p_value"}},
    {
        "evidence_type": "table",
        "label": {"table_name": "disaggregated_performance"},
        "sensitive_features": ["profession", "gender"],
    },
]
ASSESSMENT_PLAN_URL = f"http://api.credo.ai/api/v2/credoai/use_cases/${USE_CASE_ID}/assessment_plans/{POLICY_PACK_ID}"
ASSESSMENT_PLAN = {
    "use_case_id": USE_CASE_ID,
    "policy_pack_id": POLICY_PACK_ID,
    "evidence_requirements": EVIDENCE_REQUIREMENTS,
}

ASSESSMENT_PLAN_JSON_STR = json.dumps(
    {
        "data": {"attributes": ASSESSMENT_PLAN},
        "id": "id",
        "type": "assessment_plan",
    }
)


def build_metric_evidence(type):
    return MetricEvidence(
        type=type,
        value=0.2,
        model_name="superich detector",
        dataset_name="account data",
    )


def build_table_evidence(table_name):
    return TableEvidence(
        name=table_name, table_data=DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    )


class TestGovernance:
    @pytest.fixture()
    def client(self, mocker):
        # Mocking client to not send actual request to the server
        mocker.patch.object(CredoApiClient, "get")
        mocker.patch.object(CredoApiClient, "post")
        mocker.patch.object(CredoApiClient, "patch")
        mocker.patch.object(CredoApiClient, "delete")
        mocker.patch.object(CredoApiClient, "refresh_token")

        return CredoApiClient()

    @pytest.fixture()
    def api(self, mocker):
        # Mocking api to simulate result
        mocker.patch.object(CredoApi, "get_assessment_plan")
        mocker.patch.object(CredoApi, "get_assessment_plan_url")
        mocker.patch.object(CredoApi, "create_assessment")

        return CredoApi()

    @pytest.fixture()
    def gov(self, client):
        return Governance(client)

    @pytest.fixture
    def plan_file_mock(self, mocker):
        data = mocker.mock_open(read_data=ASSESSMENT_PLAN_JSON_STR)
        mocker.patch("builtins.open", data)

    # def test_real(self):
    #     client = CredoApiClient()
    #     gov = Governance(credo_api_client=client)

    #     gov.register(
    #         assessment_plan_url="http://localhost:4000/api/v2/credoai/use_cases/64YUaLWSviHgibJaRWr3ZE/assessment_plans/NYCE+1"
    #     )
    #     print(gov._use_case_id)

    def test_register_with_assessment_plan_url(self, gov, api):
        # mocking api result
        api.get_assessment_plan.return_value = ASSESSMENT_PLAN

        gov.register(assessment_plan_url=ASSESSMENT_PLAN_URL)

        api.get_assessment_plan.assert_called_with(ASSESSMENT_PLAN_URL)

        assert USE_CASE_ID == gov._use_case_id
        assert POLICY_PACK_ID == gov._policy_pack_id
        assert 3 == len(gov.get_evidence_requirements())
        assert True == gov.registered

    def test_register_with_use_case_name_and_policy_pack_key(self, gov, api):
        # mocking api result
        api.get_assessment_plan_url.return_value = ASSESSMENT_PLAN_URL
        api.get_assessment_plan.return_value = ASSESSMENT_PLAN

        gov.register(use_case_name="Fraud Detection", policy_pack_key="FAIR+1")

        api.get_assessment_plan_url.assert_called_with("Fraud Detection", "FAIR+1")
        api.get_assessment_plan.assert_called_with(ASSESSMENT_PLAN_URL)

        assert USE_CASE_ID == gov._use_case_id
        assert POLICY_PACK_ID == gov._policy_pack_id
        assert 3 == len(gov.get_evidence_requirements())
        assert True == gov.registered

    def test_register_with_use_case_name_and_policy_pack_key_not_exist(self, gov, api):
        api.get_assessment_plan_url.return_value = None

        gov.register(use_case_name="Fraud Detection", policy_pack_key="FAIR+1")

        assert False == gov.registered

    def test_register_with_assessment_plan(self, gov):

        gov.register(assessment_plan=ASSESSMENT_PLAN_JSON_STR)

        assert USE_CASE_ID == gov._use_case_id
        assert POLICY_PACK_ID == gov._policy_pack_id
        assert True == gov.registered

        req = gov._evidence_requirements[0]
        assert "metric" == req.evidence_type
        assert {"metric_type": "accuracy_score"} == req.label

    def test_register_with_assessment_plan_file(self, gov, plan_file_mock):

        gov.register(assessment_plan_file="filename")

        assert USE_CASE_ID == gov._use_case_id
        assert POLICY_PACK_ID == gov._policy_pack_id
        assert True == gov.registered

        req = gov._evidence_requirements[2]
        assert "table" == req.evidence_type
        assert {"data_type": "disaggregated_performance"} == req.label

    def test_add_evidences(self, gov):
        gov.set_evidence([build_metric_evidence("recall")])
        gov.add_evidence([build_metric_evidence("precision")])

        assert 2 == len(gov._evidences)

        gov.set_evidence([])
        assert 0 == len(gov._evidences)

    def test_export_without_registeration(self, gov):
        assert False == gov.export()

    def test_export_without_evidences(self, gov):
        gov.register(assessment_plan=ASSESSMENT_PLAN_JSON_STR)
        assert False == gov.export()

    def test_export_all_requirements_fulfilled(self, gov):
        gov.register(assessment_plan=ASSESSMENT_PLAN_JSON_STR)
        evidences = [
            build_metric_evidence("accuracy_score"),
            build_metric_evidence("p_value"),
            build_table_evidence("disaggregated_performance"),
        ]
        gov.add_evidence(evidences)
        assert True == gov.export()

    def test_export_partial_requirements_fulfilled(self, gov):
        gov.register(assessment_plan=ASSESSMENT_PLAN_JSON_STR)
        evidences = [
            build_metric_evidence("accuracy_score"),
        ]
        gov.add_evidence(evidences)
        assert False == gov.export()

    def test_export_to_file_all_requirements_fulfilled(self, gov):
        gov.register(assessment_plan=ASSESSMENT_PLAN_JSON_STR)
        evidences = [
            build_metric_evidence("accuracy_score"),
            build_metric_evidence("p_value"),
            build_table_evidence("disaggregated_performance"),
        ]
        gov.add_evidence(evidences)
        with tempfile.TemporaryDirectory() as tempDir:
            filename = f"{tempDir}/assessment.json"
            assert True == gov.export(filename)

    def test_export_to_file_partial_requirements_fulfilled(self, gov):
        gov.register(assessment_plan=ASSESSMENT_PLAN_JSON_STR)
        evidences = [
            build_metric_evidence("accuracy_score"),
        ]
        gov.add_evidence(evidences)
        with tempfile.TemporaryDirectory() as tempDir:
            filename = f"{tempDir}/assessment.json"
            assert False == gov.export(filename)
