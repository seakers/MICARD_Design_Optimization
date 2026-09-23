"""Session management + traceability logging.

Creates a per-session datestring directory and provides structured logging of
designs, prompts, LLM responses, intermediate optimizer results, and model/
version metadata, per the traceability requirements [MICARD 5.0].
"""
import json
from datetime import datetime
from pathlib import Path

from outputs.urdf_export import export_urdf
from outputs.bom import export_bom
from outputs.visualization import export_viewer


class Session:
    def __init__(self, root="designs", session_id=None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root = Path(root) / self.session_id
        # Subfolders reflect the required output/traceability artifacts [MICARD 4.0, 5.0].
        self.dirs = {
            "designs": self.root / "designs",
            "prompts": self.root / "prompts",          # LLM prompts [5.0.2]
            "responses": self.root / "responses",       # LLM responses [5.0.3]
            "intermediate": self.root / "intermediate", # optimizer intermediates [5.0.4]
            "outputs": self.root / "outputs",           # URDF/SRDF/BOM/etc. [4.0]
            "meta": self.root / "meta",                 # model/version info [5.0.5]
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self._log_meta({"session_id": self.session_id,
                        "created": datetime.now().isoformat()})

    def _timestamp(self):
        return datetime.now().strftime("%H%M%S_%f")

    def save_design(self, design, algorithm, name=None):
        """Persist a Design (complete or partial) for future reference [5.0.1]."""
        name = name or f"design_{self._timestamp()}"
        path = self.dirs["designs"] / f"{name}.json"
        path.write_text(json.dumps(design.to_dict(algorithm), indent=2), encoding="utf-8")
        urdf_path, srdf_path = export_urdf(design, self.dirs["designs"], name)
        bom_json_path, bom_csv_path = export_bom(design, self.dirs["designs"], name)
        viewer_path = export_viewer(design, self.dirs["designs"], name)
        return name

    def log_prompt(self, text, meta=None):
        """Save a prompt sent to an LLM [5.0.2]."""
        return self._write_json(self.dirs["prompts"],
                                {"prompt": text, "meta": meta or {}})

    def log_response(self, text, model=None, version=None, meta=None):
        """Save an LLM response with model/version metadata [5.0.3][5.0.5]."""
        return self._write_json(self.dirs["responses"],
                                {"response": text, "model": model,
                                 "version": version, "meta": meta or {}})

    def log_intermediate(self, name, data):
        """Save an intermediate optimizer/LLM result [5.0.4]."""
        return self._write_json(self.dirs["intermediate"], data, prefix=name)

    def _log_meta(self, data):
        return self._write_json(self.dirs["meta"], data, prefix="meta")

    def _write_json(self, folder, data, prefix="entry"):
        path = folder / f"{prefix}_{self._timestamp()}.json"
        payload = {"timestamp": datetime.now().isoformat(), **data}
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path