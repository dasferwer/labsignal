import json
import os
import time
import uuid
from datetime import UTC, datetime, timedelta

import httpx

base = os.environ.get("API_URL", "http://localhost:8098")
now = datetime.now(UTC)
end = now + timedelta(seconds=10)
with httpx.Client(base_url=base, headers={"X-API-Key": "local-demo-key"}, timeout=30) as client:
    response = client.post(
        "/experiments",
        json={
            "namespace": "demo-" + uuid.uuid4().hex[:8],
            "starts_at": (now + timedelta(seconds=1)).isoformat(),
            "ends_at": end.isoformat(),
            "outcome_seconds": 3,
        },
    )
    response.raise_for_status()
    identity = response.json()["id"]
    time.sleep(max(0, (now + timedelta(seconds=1.1) - datetime.now(UTC)).total_seconds()))
    counts = {"A": 0, "B": 0}
    index = 0
    while min(counts.values()) < 2:
        user = f"user-{index}"
        variant = client.post(f"/experiments/{identity}/assign", json={"user_id": user}).json()[
            "variant"
        ]
        counts[variant] += 1
        client.post(
            f"/experiments/{identity}/expose",
            json={
                "user_id": user,
                "pre_value": 0,
                "pre_period_end": (now - timedelta(days=1)).isoformat(),
            },
        ).raise_for_status()
        client.post(
            f"/experiments/{identity}/metrics",
            json={"user_id": user, "id": str(uuid.uuid4()), "amount": 10},
        ).raise_for_status()
        index += 1
        assert index < 100
    assert client.get(f"/experiments/{identity}/results").json()["status"] == "collecting"
    time.sleep(max(0, (end + timedelta(seconds=3.2) - datetime.now(UTC)).total_seconds()))
    report = client.get(f"/experiments/{identity}/results").json()
    assert report["status"] == "analyzed"
    assert report["n_a"] == counts["A"] and report["n_b"] == counts["B"]
    assert not report["conversion"]["significant"]
    print(
        json.dumps(
            {
                "experiment": identity,
                "exposures": counts,
                "report_status": report["status"],
                "premature_inference_blocked": True,
            },
            indent=2,
        )
    )
