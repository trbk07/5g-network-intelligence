"""Dashboard transport: no independent feature, prediction or risk calculations."""

import httpx


class BackendError(RuntimeError):
    """A concise user-facing API failure, with internal traces kept out of the UI."""


class ApiBackend:
    def __init__(self, url="http://127.0.0.1:8000"):
        self.url = url.rstrip("/")

    def request(self, method, path, **kwargs):
        try:
            response = httpx.request(method, self.url + path, timeout=20, **kwargs)
        except httpx.HTTPError:
            raise BackendError(
                "Replay API is unavailable. Check that the API and verified demo assets are ready."
            ) from None
        if response.status_code != 200:
            message = {
                404: "Selected beam is not in the loaded subset.",
                422: "Invalid origin or beam selection.",
                503: "Verified model or replay data is unavailable.",
            }.get(response.status_code, "Replay request failed.")
            raise BackendError(message)
        return response.json()

    def meta(self):
        return self.request("GET", "/replay/meta")

    def info(self):
        return self.request("GET", "/model-info")

    def forecast(self, beam_ids, origin):
        return self.request("POST", "/forecast", json={"beam_ids": beam_ids, "origin": origin})

    def risk(self, beam_ids, origin):
        return self.request("POST", "/risk", json={"beam_ids": beam_ids, "origin": origin})
