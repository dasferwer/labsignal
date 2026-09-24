import uuid
from datetime import UTC, datetime, timedelta

from labsignal.db import connect


def create(client, **changes):
    now = datetime.now(UTC)
    body = {
        "namespace": str(uuid.uuid4()),
        "starts_at": (now - timedelta(seconds=1)).isoformat(),
        "ends_at": (now + timedelta(hours=1)).isoformat(),
        "outcome_seconds": 60,
        **changes,
    }
    response = client.post("/experiments", json=body)
    assert response.status_code == 200
    return response.json()["id"], body


def test_assignment_exposure_and_event_deduplication(client):
    identity, body = create(client)
    url = f"/experiments/{identity}"
    user = {"user_id": "u1"}
    assert (
        client.post(url + "/assign", json=user).json()
        == client.post(url + "/assign", json=user).json()
    )
    metric = {**user, "id": str(uuid.uuid4()), "amount": 10}
    assert client.post(url + "/metrics", json=metric).status_code == 409
    exposure = {**user, "pre_value": 2, "pre_period_end": body["starts_at"]}
    assert client.post(url + "/expose", json=exposure).status_code == 200
    assert client.post(url + "/metrics", json=metric).json() == {"duplicate": False}
    assert client.post(url + "/metrics", json=metric).json() == {"duplicate": True}
    assert client.post(url + "/metrics", json={**metric, "amount": 20}).status_code == 409
    assert client.get(url + "/results").json()["status"] == "collecting"


def test_namespace_overlap_and_future_covariate(client):
    identity, body = create(client)
    assert client.post("/experiments", json=body).status_code == 409
    url = f"/experiments/{identity}"
    client.post(url + "/assign", json={"user_id": "u1"})
    response = client.post(
        url + "/expose", json={"user_id": "u1", "pre_value": 1, "pre_period_end": body["ends_at"]}
    )
    assert response.status_code == 422


def seed(conn, identity, n=4):
    for variant in ["A", "B"]:
        for i in range(n):
            conn.execute(
                "INSERT INTO assignments VALUES (%s,%s,%s,now()-interval '2 minutes',0)",
                (identity, f"{variant}-{i}", variant),
            )


def test_final_report_frozen_and_late_outcome_rejected(client):
    identity, _ = create(client)
    with connect() as conn:
        seed(conn, identity)
        conn.execute(
            "UPDATE experiments SET starts_at=now()-interval '1 hour',ends_at=now()-interval '3 minutes' WHERE id=%s",
            (identity,),
        )
    url = f"/experiments/{identity}"
    report = client.get(url + "/results").json()
    assert report["status"] == "analyzed"
    assert report["n_a"] == report["n_b"] == 4
    assert client.get(url + "/results").json() == report
    assert (
        client.post(
            url + "/metrics", json={"user_id": "A-0", "id": str(uuid.uuid4()), "amount": 20}
        ).status_code
        == 409
    )


def test_planned_looks_have_prespecified_alpha_and_samples(client):
    identity, _ = create(client, mode="planned", looks=[2, 4, 8])
    with connect() as conn:
        seed(conn, identity, 3)
    report = client.get(f"/experiments/{identity}/results").json()
    assert report["look"] == 0
    assert report["alpha"] == 0.05 / 3
    assert report["n_a"] == report["n_b"] == 2
    assert client.get(f"/experiments/{identity}/results").json() == report
