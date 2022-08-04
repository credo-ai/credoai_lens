"""
Test Credo API client
"""

import os
import pathlib
import pytest
import responses
from credoai.utils.credo_api_client import CredoApiClient, CredoApiConfig


class TestCredoApiConfig:
    def test_credo_api_config(self):
        config = CredoApiConfig(
            api_key="API_KEY", tenant="credo", api_server="http://localhost:4000"
        )

        assert "API_KEY" == config.api_key
        assert "credo" == config.tenant
        assert "http://localhost:4000" == config.api_server
        assert "http://localhost:4000/api/v1/credo" == config.api_base

    def test_default_config_path(self):
        path = CredoApiConfig.default_config_path()
        assert path.endswith(".credoconfig")

    def test_load_config(self):
        # Assumes that .credoconfig file exists in the same directory
        path = pathlib.Path(__file__).parent.resolve()
        config_path = os.path.join(path, ".credoconfig")

        config = CredoApiConfig()
        config.load_config(config_path)

        assert "AbcdeF" == config.api_key
        assert "credoai" == config.tenant
        assert "http://localhost:4000" == config.api_server
        assert "http://localhost:4000/api/v1/credoai" == config.api_base

    def test_valid(self):
        config = CredoApiConfig()
        assert False == config.valid

        config.load_config("/not_exist/file")
        assert False == config.valid

        config = CredoApiConfig(
            api_key="API_KEY", tenant="credo", api_server="http://localhost:4000"
        )
        assert True == config.valid


API_KEY = "API_KEY"
API_SERVER = "http://api.server"
TENANT = "credoai"


class TestCredoApiClient:
    @pytest.fixture()
    @responses.activate
    def client(self):
        responses.post(
            f"{API_SERVER}/auth/exchange", json={"access_token": "VALID_TOKEN"},
        )

        config = CredoApiConfig(api_key=API_KEY, api_server=API_SERVER, tenant=TENANT)
        return CredoApiClient(config=config)

    @responses.activate
    def test_get_request(self, client):
        responses.get(
            f"{API_SERVER}/api/v1/{TENANT}/models",
            json={
                "data": [
                    {"attributes": {"name": "model 1"}, "type": "models", "id": "123"}
                ]
            },
        )

        response = client.get("models")
        assert 1 == len(response)
        model = response[0]
        assert "model 1" == model["name"]

    @responses.activate
    def test_refresh_token(self, client):
        responses.post(
            f"{API_SERVER}/auth/exchange",
            json={"access_token": "REFRESHED_VALID_TOKEN"},
        )

        client.refresh_token()

        headers = client._session.headers
        assert "Bearer REFRESHED_VALID_TOKEN" == headers.get("Authorization")
        assert "application/vnd.api+json" == headers.get("content-type")

    def test_refresh_token_with_invalid_config(self):
        responses.post(
            f"{API_SERVER}/auth/exchange",
            json={"access_token": "REFRESHED_VALID_TOKEN"},
        )

        client = CredoApiClient(config=CredoApiConfig())
        client.refresh_token()

        headers = client._session.headers
        assert None == headers.get("Authorization")

    @responses.activate
    def test_access_token_expired(self, client):
        client.set_access_token("INVALID_TOKEN")

        # return 401 with invalid token
        responses.get(
            f"{API_SERVER}/api/v1/{TENANT}/models",
            headers={"Authorization": "Bearer INVALID_TOKEN"},
            status=401,
        )

        # refresh token to REFRESHED_VALID_TOKEN
        responses.post(
            f"{API_SERVER}/auth/exchange",
            json={"access_token": "REFRESHED_VALID_TOKEN"},
        )

        # response 200 with REFRESHED_VALID_TOKEN
        responses.get(
            f"{API_SERVER}/api/v1/{TENANT}/models",
            headers={"Authorization": "Bearer REFRESHED_VALID_TOKEN"},
            json={
                "data": [
                    {"attributes": {"name": "model 1"}, "type": "models", "id": "123"}
                ]
            },
        )

        response = client.get("models")
        headers = client._session.headers
        assert "Bearer REFRESHED_VALID_TOKEN" == headers.get("Authorization")
        assert 1 == len(response)

    @responses.activate
    def test_post_request(self, client):
        responses.post(
            f"{API_SERVER}/api/v1/{TENANT}/models",
            json={
                "data": {
                    "attributes": {"name": "model 1"},
                    "type": "models",
                    "id": "123",
                }
            },
        )

        response = client.post("models", {"name": "model 1", "$type": "models"})
        assert "model 1" == response["name"]

    @responses.activate
    def test_patch_request(self, client):
        responses.patch(
            f"{API_SERVER}/api/v1/{TENANT}/models",
            json={
                "data": {
                    "attributes": {"name": "model 1"},
                    "type": "models",
                    "id": "123",
                }
            },
        )

        response = client.patch(
            "models", {"name": "model 1", "$type": "models", "id": "123"}
        )
        assert "model 1" == response["name"]

    @responses.activate
    def test_delete_request(self, client):
        responses.delete(f"{API_SERVER}/api/v1/{TENANT}/models/123",)

        response = client.delete("models/123")
        assert None == response
