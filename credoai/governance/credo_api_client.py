"""
Defines Credo API client
"""

import os
from typing import Dict
import requests
from dotenv import dotenv_values
from json_api_doc import deserialize, serialize
from credoai.utils.common import json_dumps
from credoai.utils.constants import CREDO_URL


class CredoApiConfig:
    """
    Defines Credo API configs
    """

    def __init__(self, api_key: str = None, tenant: str = None, api_server: str = None):
        self._api_key = api_key
        self._tenant = tenant
        self._api_server = api_server
        self._api_base = self.__build_api_base()

    @staticmethod
    def default_config_path():
        """
        Returns default config path
        """
        return os.path.join(os.path.expanduser("~"), ".credoconfig")

    @property
    def api_key(self):
        """
        Returns api_key
        """
        return self._api_key

    @property
    def api_server(self):
        """
        Returns api_server
        """
        return self._api_server

    @property
    def tenant(self):
        """
        Returns tenant
        """
        return self._tenant

    @property
    def api_base(self):
        """
        Returns api_base, which is base url for API request
        """
        return self._api_base

    @property
    def valid(self):
        return bool(self._api_key) and bool(self._api_server)

    def load_config(self, config_path=None):
        """
        Load API configs from the config_path
        """
        config_path = config_path or self.default_config_path()

        if not os.path.exists(config_path):
            # return example
            self._api_key = None
            self._tenant = None
            self._api_server = None
        else:
            config = dotenv_values(config_path)
            self._tenant = config["TENANT"]
            self._api_key = config["API_KEY"]
            self._api_server = self.__build_api_server(config)
            self._api_base = self.__build_api_base()

    def __build_api_server(self, config):
        if config.get("API_URL"):
            return config["API_URL"].replace("/api/v1", "")

        return os.path.join(config.get("CREDO_URL", CREDO_URL))

    def __build_api_base(self):
        if self._api_server:
            return os.path.join(self._api_server, "api/v1", self._tenant)
        return None


class CredoApiClient:
    """
    CredoApiClient is interface class to the Credo API server.
    """

    def __init__(self, config: CredoApiConfig = None):
        if config:
            self._config = config
        else:
            self._config = CredoApiConfig()
            self._config.load_config()

        self._session = requests.Session()
        self.refresh_token()

    def refresh_token(self):
        """
        Get access token and set to headers
        """
        if self._config.valid:
            data = {"api_token": self._config.api_key, "tenant": self._config.tenant}
            headers = {"content-type": "application/json", "charset": "utf-8"}
            auth_url = os.path.join(self._config.api_server, "auth", "exchange")
            response = requests.post(auth_url, json=data, headers=headers)
            access_token = response.json()["access_token"]
            self.set_access_token(access_token)

    def set_access_token(self, access_token):
        """
        Set access token to headers
        """
        headers = {
            "Authorization": f"Bearer {access_token}",
            "accept": "application/vnd.api+json",
            "content-type": "application/vnd.api+json",
        }
        self._session.headers.update(headers)

    def __make_request(self, method, path, **kwargs):
        endpoint = self.__build_endpoint(path)
        response = self._session.request(method, endpoint, **kwargs)
        if response.status_code == 401:
            self.refresh_token()
            response = self._session.request(method, endpoint, **kwargs)

        response.raise_for_status()

        if response.content:
            return deserialize(response.json())
        else:
            return None

    def __build_endpoint(self, path):
        return os.path.join(self._config.api_base, path)

    def get(self, path: str, **kwargs):
        """
        Send get request and return retult
        """
        return self.__make_request("get", path, **kwargs)

    def post(self, path: str, data: Dict = None, **kwargs):
        """
        Send post request and return retult
        """
        data = json_dumps(serialize(data))
        return self.__make_request("post", path, data=data, **kwargs)

    def patch(self, path: str, data: Dict = None, **kwargs):
        """
        Send patch request and return retult
        """
        data = json_dumps(serialize(data))
        return self.__make_request("patch", path, data=data, **kwargs)

    def delete(self, path: str, **kwargs):
        """
        Send delete request and return retult
        """
        return self.__make_request("delete", path, **kwargs)
