import hashlib
import json
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Literal

import numpy as np
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
    pilot_id: uuid.UUID | None = None
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
        theta = body.theta
        if body.pilot_id:
            pilot = conn.execute("SELECT * FROM pilots WHERE id=%s", (body.pilot_id,)).fetchone()
            if pilot is None or pilot["ended_at"] > body.starts_at:
                raise HTTPException(
                    422, "Пилот должен существовать и завершиться до начала эксперимента"
                )
            if body.theta != 0:
                raise HTTPException(422, "При выборе пилота theta рассчитывается сервисом")
            theta = pilot["theta"]
        elif body.theta != 0:
            raise HTTPException(422, "Для CUPED нужен независимый пилот вместо произвольного theta")
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
                theta,
                secrets.token_hex(16),
            ),
        )
        protocol = {
            **body.model_dump(mode="json"),
            "theta": theta,
            "assignment": "sha256-50-50",
            "metrics": ["conversion", "revenue_cuped"],
            "multiplicity": "Holm; alpha/K для planned",
            "outcome": "один пользователь — одно наблюдение; учитываются фактические показы",
        }
        digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        prospective = body.starts_at >= datetime.now(UTC)
        conn.execute(
            "INSERT INTO protocols(experiment,snapshot,digest,prospective) VALUES (%s,%s,%s,%s)",
            (identity, Jsonb(protocol), digest, prospective),
        )
    return {"id": identity, "protocol_sha256": digest, "prospective": prospective}


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
        if assignment["exposed_at"] and (
            assignment["pre_value"] != body.pre_value
            or assignment["pre_period_end"] is not None
            and assignment["pre_period_end"] != body.pre_period_end
        ):
            raise HTTPException(409, "Показ уже зафиксирован с другой ковариатой")
        conn.execute(
            "UPDATE assignments SET exposed_at=COALESCE(exposed_at,%s),pre_value=%s,pre_period_end=%s WHERE experiment=%s AND user_id=%s",
            (now, body.pre_value, body.pre_period_end, identity, body.user_id),
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
        conn.execute("SELECT pg_advisory_xact_lock(380030,%s)", (body.id.int % (2**31 - 1),))
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
        conn.execute("SELECT pg_advisory_xact_lock(380031,%s)", (identity.int % (2**31 - 1),))
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
        protocol = conn.execute(
            "SELECT digest,prospective FROM protocols WHERE experiment=%s", (identity,)
        ).fetchone()
        report["protocol_sha256"] = protocol["digest"] if protocol else None
        report["prospective"] = bool(protocol and protocol["prospective"])
        if not report["prospective"]:
            report["decision_note"] = (
                "Ретроспективная регистрация: только описательный отчёт, подтверждающий вывод заблокирован"
            )
            report["conversion"]["significant"] = False
            report["revenue_cuped"]["significant"] = False
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


class PilotRow(BaseModel):
    pre_value: float = Field(ge=0, le=1e9, allow_inf_nan=False)
    revenue: float = Field(ge=0, le=1e9, allow_inf_nan=False)


class Pilot(BaseModel):
    ended_at: datetime
    rows: list[PilotRow] = Field(min_length=20, max_length=10000)


@app.post("/pilots")
def pilot(body: Pilot):
    if body.ended_at.tzinfo is None or body.ended_at > datetime.now(UTC):
        raise HTTPException(422, "Пилот должен завершиться в прошлом")
    x = np.array([r.pre_value for r in body.rows])
    y = np.array([r.revenue for r in body.rows])
    if np.var(x) == 0:
        raise HTTPException(422, "В пилоте нет вариации ковариаты")
    theta = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
    if not np.isfinite(theta) or abs(theta) > 100:
        raise HTTPException(422, "Нестабильный коэффициент CUPED")
    snapshot = body.model_dump(mode="json")
    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    identity = uuid.uuid4()
    with connect() as conn:
        conn.execute(
            "INSERT INTO pilots(id,ended_at,snapshot,digest,theta) VALUES (%s,%s,%s,%s,%s)",
            (identity, body.ended_at, Jsonb(snapshot), digest, theta),
        )
    return {"id": identity, "theta": theta, "sha256": digest, "observations": len(body.rows)}


@app.get("/experiments/{identity}/protocol")
def protocol(identity: uuid.UUID):
    with connect() as conn:
        row = conn.execute(
            "SELECT snapshot,digest,prospective,created_at FROM protocols WHERE experiment=%s",
            (identity,),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "Протокол не найден для этого запуска")
    return row
