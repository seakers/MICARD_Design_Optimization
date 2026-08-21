"""Thin, optional client for the micard-metrics service (Unix domain socket).

Wraps the trac_ik-backed metrics service, which computes conditioning metrics
(w, k_inv, k_min, k, alpha, beta, cosine, limit_penalty) plus IK-backed
reachability from a URDF and a base/tip link [micard-metrics README]. Batches
every call ("batch, always") and degrades gracefully when the service is not
reachable so the prototype still runs on the placeholder estimators [MICARD 5.0].
"""
import json
import socket
from http.client import HTTPConnection


class _UnixHTTPConnection(HTTPConnection):
    """HTTPConnection over an AF_UNIX socket (the service exposes no TCP port)."""

    def __init__(self, socket_path, timeout=30):
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self._socket_path)
        self.sock = s


class MetricsClient:
    """Minimal client; None-returns/raises are handled by callers for fallback."""

    def __init__(self, socket_path="run/api.sock", timeout=30):
        self.socket_path = socket_path
        self.timeout = timeout

    def available(self):
        """True if the service answers /health, else False (enables fallback)."""
        try:
            self._request("GET", "/health")
            return True
        except Exception:
            return False

    def _request(self, method, path, body=None):
        conn = _UnixHTTPConnection(self.socket_path, timeout=self.timeout)
        try:
            data = json.dumps(body).encode("utf-8") if body is not None else None
            headers = {"Content-Type": "application/json"} if data else {}
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8")
            if resp.status >= 400:
                raise RuntimeError(f"{method} {path} -> {resp.status}: {raw}")
            return json.loads(raw) if raw else {}
        finally:
            conn.close()

    def register_chain(self, urdf_xml, base_link, tip_link,
                       srdf=None, collision=False, **kwargs):
        """Register a chain and return the response.

        Passes URDF as inline XML (the README's portable option, avoiding mesh
        mounts — our tube links are primitive cylinders). Registration is
        idempotent, keyed by URDF content + settings [micard-metrics README].
        Surfaces always_colliding_pairs / missing_collision_geometry so a bad
        SRDF is caught immediately rather than as "everything unreachable".
        """
        body = {"urdf": urdf_xml, "base_link": base_link, "tip_link": tip_link,
                "collision": collision}
        if srdf is not None:
            body["srdf"] = srdf
        body.update(kwargs)
        return self._request("POST", "/chains", body)

    def metrics(self, chain_id, configurations, task_dir=None):
        """Every conditioning metric per configuration, in ONE batched call."""
        body = {"configurations": configurations}
        if task_dir is not None:
            body["task_dir"] = task_dir
        return self._request("POST", f"/chains/{chain_id}/metrics", body)

    def optimize_at_poses(self, chain_id, objective, poses, seed=0, task_dir=None):
        """Batched reachability + best-conditioned config per pose (one request).

        The README recommends this for 'which of these poses can the arm reach,
        and how well' — unreachable poses are flagged, not fatal.
        """
        body = {"objective": objective, "seed": seed, "poses": poses}
        if task_dir is not None:
            body["task_dir"] = task_dir
        return self._request("POST", f"/chains/{chain_id}/optimize_at_poses", body)