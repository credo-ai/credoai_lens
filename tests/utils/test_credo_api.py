import pytest
from credoai.utils.credo_api import CredoApi
from credoai.utils.credo_api_client import CredoApiClient


class TestCredoApi:

    @pytest.fixture()
    def client(self, mocker):
        # Use mock for CredoApiClient becasue we do not want to call HTTP request from test.
        # We will only check if CredoApi is calling client method with right arguments.
        mocker.patch.object(CredoApiClient, "get")
        mocker.patch.object(CredoApiClient, "post")
        mocker.patch.object(CredoApiClient, "patch")
        mocker.patch.object(CredoApiClient, "delete")
        return CredoApiClient()

    @pytest.fixture()
    def api(self, client):
        return CredoApi(client=client)

    # def test_real(self):
    #     client = CredoApiClient()
    #     api = CredoApi(client=client)
    #     response = api.get_assessment("wcTP5F6URXGpUz4JTC9o7Y")
    #     print(response)

    def test_get_assessment(self, api, client):
        api.get_assessment("ASESS_ID")

        client.get.assert_called_with("use_case_assessments/ASESS_ID")

    def test_create_assessment(self, api, client):
        data = {"reports": [], "metrics": []}
        api.create_assessment("USE_CASE_ID", "MODEL_ID", data)

        client.post.assert_called_with(
            "use_cases/USE_CASE_ID/models/MODEL_ID/assessments", data)

    def test_get_assessment_spec(self, api, client):
        spec_url = "use_cases/USE_CASE_ID/models/MODEL_ID/assessment"
        api.get_assessment_spec(spec_url)
        client.get.assert_called_with(spec_url)

    def test_get_dataset_by_name(self, api, client):
        name = "dataset name"
        client.get.return_value = [{"id": "123", "name": name}]

        response = api.get_dataset_by_name(name)
        assert name == response["name"]

        client.get.assert_called_with(
            "datasets", params={'filter[name]': name})

    def test_get_dataset_by_name_return_none(self, api, client):
        name = "dataset name"
        client.get.return_value = []

        response = api.get_dataset_by_name(name)
        assert None == response

    def test_get_model_by_name(self, api, client):
        name = "model name"
        client.get.return_value = [{"id": "123", "name": name}]

        response = api.get_model_by_name(name)
        assert name == response["name"]

        client.get.assert_called_with(
            "models", params={'filter[name]': name})

    def test_get_model_by_name_return_none(self, api, client):
        name = "model name"
        client.get.return_value = []

        response = api.get_model_by_name(name)
        assert None == response

    def test_get_use_case_by_name(self, api, client):
        name = "use_case name"
        client.get.return_value = [{"id": "123", "name": name}]

        response = api.get_use_case_by_name(name)
        assert name == response["name"]

        client.get.assert_called_with(
            "use_cases", params={'filter[name]': name})

    def test_get_use_case_by_name_return_none(self, api, client):
        name = "use_case name"
        client.get.return_value = []

        response = api.get_use_case_by_name(name)
        assert None == response

    def test_register_dataset_returns_exiting(self, api, client):
        name = "dataset name"
        client.get.return_value = [{"id": "123", "name": name}]

        response = api.register_dataset(name)
        assert name == response["name"]
        client.get.assert_called_with(
            "datasets", params={'filter[name]': name})

    def test_register_dataset_create_new(self, api, client):
        name = "dataset name"
        client.get.return_value = []
        client.post.return_value = {"id": "123", "name": name}

        response = api.register_dataset(name)
        assert name == response["name"]
        client.post.assert_called_with(
            "datasets", {'name': name, '$type': 'datasets'})

    def test_register_model_returns_exiting(self, api, client):
        name = "model name"
        client.get.return_value = [{"id": "123", "name": name}]

        response = api.register_model(name)
        assert name == response["name"]
        client.get.assert_called_with(
            "models", params={'filter[name]': name})

    def test_register_model_create_new(self, api, client):
        name = "model name"
        client.get.return_value = []
        client.post.return_value = {"id": "123", "name": name}

        response = api.register_model(name)
        assert name == response["name"]
        client.post.assert_called_with(
            "models", {'name': name, 'version': '1.0', '$type': 'models'})

    def test_register_model_to_usecase(self, api, client):
        api.register_model_to_usecase("USE_CASE_ID", "MODEL_ID")
        client.post.assert_called_with(
            'use_cases/USE_CASE_ID/relationships/models', [{'id': 'MODEL_ID', '$type': 'models'}])

    def test_register_dataset_to_model(self, api, client):
        api.register_dataset_to_model("MODEL_ID", "DATASET_ID")
        client.patch.assert_called_with(
            'models/MODEL_ID/relationships/dataset', {'id': 'DATASET_ID', '$type': 'datasets'})

    def test_register_dataset_to_model_usecase(self, api, client):
        api.register_dataset_to_model_usecase(
            "USE_CASE_ID", "MODEL_ID", "DATASET_ID")
        client.patch.assert_called_with(
            'use_cases/USE_CASE_ID/models/MODEL_ID/config', {'dataset_id': 'DATASET_ID', '$type': 'use_case_model_configs', 'id': 'unknown'})
