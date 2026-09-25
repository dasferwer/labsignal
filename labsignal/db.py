import os

import psycopg
from psycopg.rows import dict_row


def connect():
    return psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row)


def init():
    with connect() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(380029)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id uuid PRIMARY KEY, namespace text NOT NULL, starts_at timestamptz NOT NULL,
                ends_at timestamptz NOT NULL, outcome_seconds integer NOT NULL,
                mode text NOT NULL, looks integer[] NOT NULL, theta double precision NOT NULL,
                salt text NOT NULL);
            CREATE TABLE IF NOT EXISTS assignments (
                experiment uuid REFERENCES experiments(id), user_id text NOT NULL, variant text NOT NULL,
                exposed_at timestamptz, pre_value double precision,
                PRIMARY KEY(experiment,user_id));
            CREATE TABLE IF NOT EXISTS events (
                id uuid PRIMARY KEY, experiment uuid NOT NULL, user_id text NOT NULL,
                amount double precision NOT NULL, received_at timestamptz NOT NULL DEFAULT now(),
                FOREIGN KEY(experiment,user_id) REFERENCES assignments(experiment,user_id));
            CREATE TABLE IF NOT EXISTS reports (
                experiment uuid REFERENCES experiments(id), look integer NOT NULL, report jsonb NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(experiment,look));
            ALTER TABLE assignments ADD COLUMN IF NOT EXISTS pre_period_end timestamptz;
            CREATE TABLE IF NOT EXISTS pilots (
                id uuid PRIMARY KEY,ended_at timestamptz NOT NULL,snapshot jsonb NOT NULL,
                digest text NOT NULL,theta double precision NOT NULL,created_at timestamptz NOT NULL DEFAULT now());
            CREATE TABLE IF NOT EXISTS protocols (
                experiment uuid PRIMARY KEY REFERENCES experiments(id),snapshot jsonb NOT NULL,
                digest text NOT NULL,prospective boolean NOT NULL,created_at timestamptz NOT NULL DEFAULT now());
        """)
