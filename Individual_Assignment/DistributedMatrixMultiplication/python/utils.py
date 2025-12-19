import json
import os
import shutil
import numpy as np
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple
from urllib.error import URLError, HTTPError
from pyspark.sql import SparkSession


@dataclass(frozen=True)
class Case:
    m: int
    n: int
    p: int


@dataclass
class Result:
    mode: str
    m: int
    n: int
    p: int
    dtype: str
    ok: bool
    elapsed_s: float
    checksum: Optional[float]
    notes: str
    extra: Dict[str, Any]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def wipe_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def disk_free_bytes(path: str) -> int:
    return shutil.disk_usage(path).free


def dtype_np(dtype_str: str):
    if dtype_str == "float32":
        return np.float32
    if dtype_str == "float64":
        return np.float64
    raise ValueError("dtype must be 'float32' or 'float64'")


def estimate_dense_bytes(m: int, n: int, p: int, itemsize: int) -> int:
    return itemsize * (m * n + n * p + m * p)


class SafetyGuard:
    def __init__(self, max_estimated_dense_bytes: int, min_free_disk_bytes: int):
        self.max_estimated_dense_bytes = max_estimated_dense_bytes
        self.min_free_disk_bytes = min_free_disk_bytes

    def skip_dense_reason(self, m: int, n: int, p: int, dt) -> Optional[str]:
        est = estimate_dense_bytes(m, n, p, np.dtype(dt).itemsize)
        if est > self.max_estimated_dense_bytes:
            return f"Skipped: dense estimation A+B+C ≈ {est/1e9:.2f} GB exceeds the limit."
        return None

    def spark_disk_ok(self, spark_local_dir: str) -> tuple[bool, int]:
        free = disk_free_bytes(spark_local_dir)
        return (free >= self.min_free_disk_bytes), free


class JsonReporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        ensure_dir(self.output_dir)

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _json_safe(self, obj):
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)

        try:
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
        except Exception:
            pass

        try:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except Exception:
            pass

        if isinstance(obj, (set, tuple)):
            return list(obj)
        if isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._json_safe(v) for v in obj]

        return obj

    def save(self, config: Dict[str, Any], results: List[Result]) -> str:
        path = os.path.join(self.output_dir, f"dmm_results_{self._ts()}.json")
        payload = {
            "created_at": datetime.now().isoformat(),
            "config": self._json_safe(config),
            "results": [asdict(r) for r in results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path


def spark_make_session(cfg: Dict[str, Any], run_tag: str):
    spark_local_dir = cfg["spark_local_dir"]
    ensure_dir(spark_local_dir)
    builder = (
        SparkSession.builder
        .appName(f"{cfg['spark_app_name']}-{run_tag}")
        .master(cfg["spark_master"])
        .config("spark.local.dir", os.path.abspath(spark_local_dir))
        .config("spark.default.parallelism", str(cfg["spark_partitions"]))
        .config("spark.sql.shuffle.partitions", str(cfg["spark_partitions"]))
    )
    return builder.getOrCreate()


def _http_get_json(url: str, timeout_sec: float = 2.0) -> Optional[Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            return json.loads(data)
    except (URLError, HTTPError, TimeoutError, ValueError):
        return None


def spark_collect_metrics_from_ui(spark) -> Dict[str, Any]:
    sc = spark.sparkContext
    ui = sc.uiWebUrl
    if not ui:
        return {"metrics_note": "Spark UI is not available (uiWebUrl is None)."}

    base = ui.rstrip("/")
    apps = _http_get_json(f"{base}/api/v1/applications")
    if not apps or not isinstance(apps, list):
        return {"metrics_note": "Spark UI API not reachable or no applications found.", "sparkUiUrl": base}

    app_id = apps[0].get("id")
    if not app_id:
        return {"metrics_note": "Could not read application id from Spark UI.", "sparkUiUrl": base}

    stages = _http_get_json(f"{base}/api/v1/applications/{app_id}/stages")
    if stages is None or not isinstance(stages, list):
        return {"metrics_note": "Could not read stages from Spark UI.", "sparkUiUrl": base, "appId": app_id}

    metrics = {
        "sparkUiUrl": base,
        "appId": app_id,
        "stages": len(stages),
        "tasks": 0,
        "shuffleReadBytes": 0,
        "shuffleWriteBytes": 0,
        "memoryBytesSpilled": 0,
        "diskBytesSpilled": 0,
        "executorRunTimeMs": 0,
        "jvmGCTimeMs": 0,
        "resultSizeBytes": 0,
    }

    for st in stages:
        metrics["tasks"] += int(st.get("numTasks", 0) or 0)
        metrics["shuffleReadBytes"] += int(st.get("shuffleReadBytes", 0) or 0)
        metrics["shuffleWriteBytes"] += int(st.get("shuffleWriteBytes", 0) or 0)
        metrics["memoryBytesSpilled"] += int(st.get("memoryBytesSpilled", 0) or 0)
        metrics["diskBytesSpilled"] += int(st.get("diskBytesSpilled", 0) or 0)
        metrics["executorRunTimeMs"] += int(st.get("executorRunTime", 0) or 0)
        metrics["jvmGCTimeMs"] += int(st.get("jvmGcTime", 0) or 0)
        if "resultSize" in st:
            metrics["resultSizeBytes"] += int(st.get("resultSize", 0) or 0)

    return metrics
