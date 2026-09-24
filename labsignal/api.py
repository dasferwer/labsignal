import hashlib
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field, model_validator

from labsignal.db import connect, init
from labsignal.stats import analyze


@asynccontextmanager
async def lifespan(app):
    init()
    yield


def authorize(x_api_key: str = Header(default="")):
    key = os.environ.get("API_KEY", "")
    if not key or not secrets.compare_digest(key, x_api_key):
        raise HTTPException(401, "Неверный API-ключ")


app = FastAPI(title="LabSignal", lifespan=lifespan, dependencies=[Depends(authorize)])


class Experiment(BaseModel):
    namespace: str = Field(min_length=1, max_length=80)
    starts_at: datetime
    ends_at: datetime
    outcome_seconds: int = Field(default=3600, ge=1, le=86400)
    mode: Literal["fixed", "planned"] = "fixed"
    looks: list[int] = Field(default_factory=lambda: [100, 200, 400], min_length=1, max_length=5)
    theta: float = Field(default=0, ge=-100, le=100, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid(self):
        if (
            self.starts_at.tzinfo is None
            or self.ends_at.tzinfo is None
            or self.starts_at >= self.ends_at
        ):
            raise ValueError("Нужны даты с часовым поясом и конец после начала")
        if self.looks != sorted(set(self.looks)) or min(self.looks) < 2:
            raise ValueError("Размеры проверок должны строго расти и быть не меньше двух")
        return self


class User(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)


class Exposure(User):
    pre_value: float = Field(default=0, ge=0, le=1e9, allow_inf_nan=False)
    pre_period_end: datetime


class Metric(User):
    id: uuid.UUID
    amount: float = Field(ge=0, le=1e9, allow_inf_nan=False)


def get(conn, identity):
    row = conn.execute("SELECT * FROM experiments WHERE id=%s", (identity,)).fetchone()
    if not row:
        raise HTTPException(404, "Эксперимент не найден")
    return row


def active(experiment, now):
    if not experiment["starts_at"] <= now < experiment["ends_at"]:
        raise HTTPException(409, "Эксперимент сейчас не принимает показы")


@app.post("/experiments")
def create(body: Experiment):
    identity = uuid.uuid4()
    with connect() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(380028)")
        overlap = conn.execute(
            "SELECT 1 FROM experiments WHERE namespace=%s AND starts_at<%s AND ends_at>%s",
            (body.namespace, body.ends_at, body.starts_at),
        ).fetchone()
        if overlap:
            raise HTTPException(409, "В этом пространстве уже есть пересекающийся эксперимент")
        conn.execute(
            "INSERT INTO experiments VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                identity,
                body.namespace,
                body.starts_at,
                body.ends_at,
                body.outcome_seconds,
                body.mode,
                body.looks,
                body.theta,
                secrets.token_hex(16),
            ),
        )
    return {"id": identity}


@app.post("/experiments/{identity}/assign")
def assign(identity: uuid.UUID, body: User):
    with connect() as conn:
        experiment = get(conn, identity)
        active(experiment, datetime.now(UTC))
        digest = hashlib.sha256(f"{experiment['salt']}:{body.user_id}".encode()).digest()
        variant = "A" if int.from_bytes(digest[:8], "big") < 2**63 else "B"
        conn.execute(
            "INSERT INTO assignments(experiment,user_id,variant) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
            (identity, body.user_id, variant),
        )
    return {"variant": variant}


@app.post("/experiments/{identity}/expose")
def expose(identity: uuid.UUID, body: Exposure):
    with connect() as conn:
        experiment = get(conn, identity)
        now = datetime.now(UTC)
        active(experiment, now)
        if body.pre_period_end.tzinfo is None or body.pre_period_end > experiment["starts_at"]:
            raise HTTPException(422, "Ковариата должна относиться к периоду до эксперимента")
        assignment = conn.execute(
            "SELECT * FROM assignments WHERE experiment=%s AND user_id=%s FOR UPDATE",
            (identity, body.user_id),
        ).fetchone()
        if not assignment:
            raise HTTPException(409, "Сначала получите вариант")
        if assignment["exposed_at"] and assignment["pre_value"] != body.pre_value:
            raise HTTPException(409, "Показ уже зафиксирован с другой ковариатой")
        conn.execute(
            "UPDATE assignments SET exposed_at=COALESCE(exposed_at,%s),pre_value=%s WHERE experiment=%s AND user_id=%s",
            (now, body.pre_value, identity, body.user_id),
        )
    return {"variant": assignment["variant"], "exposed": True}


@app.post("/experiments/{identity}/metrics")
def metric(identity: uuid.UUID, body: Metric):
    with connect() as conn:
        experiment = get(conn, identity)
        assignment = conn.execute(
            "SELECT * FROM assignments WHERE experiment=%s AND user_id=%s FOR UPDATE",
            (identity, body.user_id),
        ).fetchone()
        old = conn.execute(
            "SELECT experiment,user_id,amount FROM events WHERE id=%s", (body.id,)
        ).fetchone()
        if old:
            if old != {"experiment": identity, "user_id": body.user_id, "amount": body.amount}:
                raise HTTPException(409, "ID события уже использован")
            return {"duplicate": True}
        if not assignment or not assignment["exposed_at"]:
            raise HTTPException(409, "Нет фактического показа")
        if datetime.now(UTC) >= assignment["exposed_at"] + timedelta(
            seconds=experiment["outcome_seconds"]
        ):
            raise HTTPException(409, "Окно результата закрыто")
        conn.execute(
            "INSERT INTO events(id,experiment,user_id,amount) VALUES (%s,%s,%s,%s)",
            (body.id, identity, body.user_id, body.amount),
        )
    return {"duplicate": False}


@app.get("/experiments/{identity}/results")
def results(identity: uuid.UUID):
    with connect() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(380028)")
        experiment = get(conn, identity)
        now = datetime.now(UTC)
        if experiment["mode"] == "fixed" and now < experiment["ends_at"] + timedelta(
            seconds=experiment["outcome_seconds"]
        ):
            return {
                "status": "collecting",
                "reason": "Фиксированный срок и окно результатов ещё не завершены",
            }
        # Дожидаемся уже начатых записей метрик до фиксации зрелой выборки.
        conn.execute(
            "SELECT user_id FROM assignments WHERE experiment=%s AND exposed_at<=%s ORDER BY user_id FOR UPDATE",
            (identity, now - timedelta(seconds=experiment["outcome_seconds"])),
        ).fetchall()
        data = conn.execute(
            """SELECT a.user_id,a.variant,a.pre_value,a.exposed_at,
            count(e.id)>0 AS converted,COALESCE(sum(e.amount),0) AS revenue
            FROM assignments a LEFT JOIN events e ON e.experiment=a.experiment AND e.user_id=a.user_id
            WHERE a.experiment=%s AND a.exposed_at<=%s
            GROUP BY a.experiment,a.user_id ORDER BY a.exposed_at,a.user_id""",
            (identity, now - timedelta(seconds=experiment["outcome_seconds"])),
        ).fetchall()
        a, b = [r for r in data if r["variant"] == "A"], [r for r in data if r["variant"] == "B"]
        if experiment["mode"] == "fixed":
            look, size, alpha = 0, None, 0.05
        else:
            eligible = [
                (i, n) for i, n in enumerate(experiment["looks"]) if min(len(a), len(b)) >= n
            ]
            if not eligible:
                return {
                    "status": "collecting",
                    "reason": "Недостаточно зрелых показов для первой проверки",
                }
            look, size = eligible[-1]
            alpha = 0.05 / len(experiment["looks"])
        previous = conn.execute(
            "SELECT report FROM reports WHERE experiment=%s AND look=%s", (identity, look)
        ).fetchone()
        if previous:
            return previous["report"]
        try:
            report = analyze(
                a[:size], b[:size], experiment["theta"], alpha, counts=(len(a), len(b))
            )
        except ValueError as exc:
            return {"status": "insufficient", "reason": str(exc)}
        report.update(
            {
                "status": "analyzed",
                "look": look,
                "mode": experiment["mode"],
                "selected_users": {
                    "A": [r["user_id"] for r in a[:size]],
                    "B": [r["user_id"] for r in b[:size]],
                },
            }
        )
        conn.execute(
            "INSERT INTO reports(experiment,look,report) VALUES (%s,%s,%s)",
            (identity, look, Jsonb(report)),
        )
        return report


@app.get("/health")
def health():
    with connect() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}
