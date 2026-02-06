from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


_DEFAULT_DB_DIR = Path.home() / ".git-xrays"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "runs.db"


class RunStore:
    """Persists --all run results to DuckDB for trend analysis."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            self._db_path = _DEFAULT_DB_PATH
        else:
            self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._db_path))
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id              VARCHAR PRIMARY KEY,
                repo_path           VARCHAR NOT NULL,
                created_at          TIMESTAMP NOT NULL,
                window_days         INTEGER NOT NULL,
                from_date           TIMESTAMP NOT NULL,
                to_date             TIMESTAMP NOT NULL,
                total_commits       INTEGER NOT NULL,
                first_commit_date   TIMESTAMP,
                last_commit_date    TIMESTAMP,
                hotspot_file_count       INTEGER NOT NULL,
                developer_risk_index     INTEGER NOT NULL,
                knowledge_island_count   INTEGER NOT NULL,
                coupling_pair_count      INTEGER NOT NULL,
                anemia_total_classes     INTEGER NOT NULL,
                anemia_anemic_count      INTEGER NOT NULL,
                anemia_anemic_pct        DOUBLE NOT NULL,
                anemia_average_ams       DOUBLE NOT NULL,
                complexity_total_functions INTEGER NOT NULL,
                complexity_avg           DOUBLE NOT NULL,
                complexity_max           INTEGER NOT NULL,
                complexity_high_count    INTEGER NOT NULL,
                clustering_k             INTEGER NOT NULL,
                clustering_silhouette    DOUBLE NOT NULL,
                effort_total_files       INTEGER NOT NULL,
                effort_model_r_squared   DOUBLE NOT NULL,
                effort_coefficients      VARCHAR NOT NULL,
                dx_score                 DOUBLE NOT NULL,
                dx_throughput            DOUBLE NOT NULL,
                dx_feedback_delay        DOUBLE NOT NULL,
                dx_focus_ratio           DOUBLE NOT NULL,
                dx_cognitive_load        DOUBLE NOT NULL,
                dx_weights               VARCHAR NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS hotspot_files (
                run_id    VARCHAR NOT NULL,
                file_path VARCHAR NOT NULL,
                change_frequency INTEGER NOT NULL,
                code_churn       INTEGER NOT NULL,
                hotspot_score    DOUBLE NOT NULL,
                rework_ratio     DOUBLE NOT NULL,
                PRIMARY KEY (run_id, file_path)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_files (
                run_id    VARCHAR NOT NULL,
                file_path VARCHAR NOT NULL,
                knowledge_concentration DOUBLE NOT NULL,
                primary_author          VARCHAR NOT NULL,
                primary_author_pct      DOUBLE NOT NULL,
                is_knowledge_island     BOOLEAN NOT NULL,
                author_count            INTEGER NOT NULL,
                PRIMARY KEY (run_id, file_path)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS coupling_pairs (
                run_id    VARCHAR NOT NULL,
                file_a    VARCHAR NOT NULL,
                file_b    VARCHAR NOT NULL,
                shared_commits    INTEGER NOT NULL,
                coupling_strength DOUBLE NOT NULL,
                support           DOUBLE NOT NULL,
                PRIMARY KEY (run_id, file_a, file_b)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS file_pain (
                run_id    VARCHAR NOT NULL,
                file_path VARCHAR NOT NULL,
                size_normalized      DOUBLE NOT NULL,
                volatility_normalized DOUBLE NOT NULL,
                distance_normalized  DOUBLE NOT NULL,
                pain_score           DOUBLE NOT NULL,
                PRIMARY KEY (run_id, file_path)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS anemia_classes (
                run_id     VARCHAR NOT NULL,
                file_path  VARCHAR NOT NULL,
                class_name VARCHAR NOT NULL,
                field_count          INTEGER NOT NULL,
                behavior_method_count INTEGER NOT NULL,
                dbsi                 DOUBLE NOT NULL,
                ams                  DOUBLE NOT NULL,
                PRIMARY KEY (run_id, file_path, class_name)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS complexity_functions (
                run_id        VARCHAR NOT NULL,
                file_path     VARCHAR NOT NULL,
                function_name VARCHAR NOT NULL,
                line_number   INTEGER NOT NULL,
                cyclomatic_complexity INTEGER NOT NULL,
                max_nesting_depth     INTEGER NOT NULL,
                length                INTEGER NOT NULL,
                PRIMARY KEY (run_id, file_path, function_name, line_number)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cluster_summaries (
                run_id     VARCHAR NOT NULL,
                cluster_id INTEGER NOT NULL,
                label      VARCHAR NOT NULL,
                size       INTEGER NOT NULL,
                centroid_file_count  DOUBLE NOT NULL,
                centroid_total_churn DOUBLE NOT NULL,
                centroid_add_ratio   DOUBLE NOT NULL,
                PRIMARY KEY (run_id, cluster_id)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cluster_drift (
                run_id        VARCHAR NOT NULL,
                cluster_label VARCHAR NOT NULL,
                first_half_pct  DOUBLE NOT NULL,
                second_half_pct DOUBLE NOT NULL,
                drift           DOUBLE NOT NULL,
                trend           VARCHAR NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS effort_files (
                run_id    VARCHAR NOT NULL,
                file_path VARCHAR NOT NULL,
                rei_score    DOUBLE NOT NULL,
                proxy_label  DOUBLE NOT NULL,
                PRIMARY KEY (run_id, file_path)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS dx_cognitive_files (
                run_id    VARCHAR NOT NULL,
                file_path VARCHAR NOT NULL,
                complexity_score   DOUBLE NOT NULL,
                coordination_score DOUBLE NOT NULL,
                knowledge_score    DOUBLE NOT NULL,
                change_rate_score  DOUBLE NOT NULL,
                composite_load     DOUBLE NOT NULL,
                PRIMARY KEY (run_id, file_path)
            )
        """)

    def save_run(
        self,
        run_id: str,
        repo_path: str,
        window_days: int,
        summary,
        hotspot,
        knowledge,
        coupling,
        anemia,
        complexity,
        clustering,
        effort,
        dx,
    ) -> None:
        """Persist all analysis results in a single transaction."""
        now = datetime.now(timezone.utc)

        self._conn.begin()
        try:
            # Insert runs row
            self._conn.execute(
                """INSERT INTO runs VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )""",
                [
                    run_id,
                    repo_path,
                    now,
                    window_days,
                    hotspot.from_date,
                    hotspot.to_date,
                    hotspot.total_commits,
                    summary.first_commit_date,
                    summary.last_commit_date,
                    # hotspot
                    len(hotspot.files),
                    # knowledge
                    knowledge.developer_risk_index,
                    knowledge.knowledge_island_count,
                    # coupling
                    len(coupling.coupling_pairs),
                    # anemia
                    anemia.total_classes,
                    anemia.anemic_count,
                    anemia.anemic_percentage,
                    anemia.average_ams,
                    # complexity
                    complexity.total_functions,
                    complexity.avg_complexity,
                    complexity.max_complexity,
                    complexity.high_complexity_count,
                    # clustering
                    clustering.k,
                    clustering.silhouette_score,
                    # effort
                    effort.total_files,
                    effort.model_r_squared,
                    json.dumps(effort.coefficients),
                    # dx
                    dx.dx_score,
                    dx.metrics.throughput,
                    dx.metrics.feedback_delay,
                    dx.metrics.focus_ratio,
                    dx.metrics.cognitive_load,
                    json.dumps(dx.weights),
                ],
            )

            # Hotspot files
            for f in hotspot.files:
                self._conn.execute(
                    "INSERT INTO hotspot_files VALUES (?, ?, ?, ?, ?, ?)",
                    [run_id, f.file_path, f.change_frequency, f.code_churn,
                     f.hotspot_score, f.rework_ratio],
                )

            # Knowledge files
            for f in knowledge.files:
                self._conn.execute(
                    "INSERT INTO knowledge_files VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [run_id, f.file_path, f.knowledge_concentration,
                     f.primary_author, f.primary_author_pct,
                     f.is_knowledge_island, f.author_count],
                )

            # Coupling pairs
            for p in coupling.coupling_pairs:
                self._conn.execute(
                    "INSERT INTO coupling_pairs VALUES (?, ?, ?, ?, ?, ?)",
                    [run_id, p.file_a, p.file_b, p.shared_commits,
                     p.coupling_strength, p.support],
                )

            # File pain
            for fp in coupling.file_pain:
                self._conn.execute(
                    "INSERT INTO file_pain VALUES (?, ?, ?, ?, ?, ?)",
                    [run_id, fp.file_path, fp.size_normalized,
                     fp.volatility_normalized, fp.distance_normalized,
                     fp.pain_score],
                )

            # Anemia classes
            for fa in anemia.files:
                for cm in fa.classes:
                    self._conn.execute(
                        "INSERT INTO anemia_classes VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [run_id, cm.file_path, cm.class_name, cm.field_count,
                         cm.behavior_method_count, cm.dbsi, cm.ams],
                    )

            # Complexity functions
            for fc in complexity.files:
                for fn in fc.functions:
                    self._conn.execute(
                        "INSERT INTO complexity_functions VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [run_id, fn.file_path, fn.function_name, fn.line_number,
                         fn.cyclomatic_complexity, fn.max_nesting_depth,
                         fn.length],
                    )

            # Cluster summaries
            for cs in clustering.clusters:
                self._conn.execute(
                    "INSERT INTO cluster_summaries VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [run_id, cs.cluster_id, cs.label, cs.size,
                     cs.centroid_file_count, cs.centroid_total_churn,
                     cs.centroid_add_ratio],
                )

            # Cluster drift
            for cd in clustering.drift:
                self._conn.execute(
                    "INSERT INTO cluster_drift VALUES (?, ?, ?, ?, ?, ?)",
                    [run_id, cd.cluster_label, cd.first_half_pct,
                     cd.second_half_pct, cd.drift, cd.trend],
                )

            # Effort files
            for ef in effort.files:
                self._conn.execute(
                    "INSERT INTO effort_files VALUES (?, ?, ?, ?)",
                    [run_id, ef.file_path, ef.rei_score, ef.proxy_label],
                )

            # DX cognitive files
            for df in dx.cognitive_load_files:
                self._conn.execute(
                    "INSERT INTO dx_cognitive_files VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [run_id, df.file_path, df.complexity_score,
                     df.coordination_score, df.knowledge_score,
                     df.change_rate_score, df.composite_load],
                )

            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def list_runs(self) -> list[dict]:
        """Return past runs ordered by created_at descending."""
        result = self._conn.execute(
            """SELECT run_id, repo_path, created_at, window_days,
                      total_commits, hotspot_file_count, dx_score
               FROM runs
               ORDER BY created_at DESC"""
        ).fetchall()
        columns = ["run_id", "repo_path", "created_at", "window_days",
                    "total_commits", "hotspot_file_count", "dx_score"]
        return [dict(zip(columns, row)) for row in result]

    def get_run(self, run_id: str) -> dict | None:
        """Return full runs row as dict, or None if not found."""
        result = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", [run_id]
        ).fetchone()
        if result is None:
            return None
        cols = [desc[0] for desc in self._conn.description]
        return dict(zip(cols, result))

    def list_repos(self) -> list[str]:
        """Return distinct repo_path values sorted alphabetically."""
        rows = self._conn.execute(
            "SELECT DISTINCT repo_path FROM runs ORDER BY repo_path"
        ).fetchall()
        return [r[0] for r in rows]

    def list_runs_for_repo(self, repo_path: str) -> list[dict]:
        """Return runs for a specific repo, ordered by created_at descending."""
        result = self._conn.execute(
            """SELECT run_id, repo_path, created_at, window_days,
                      total_commits, hotspot_file_count, dx_score
               FROM runs
               WHERE repo_path = ?
               ORDER BY created_at DESC""",
            [repo_path],
        ).fetchall()
        columns = ["run_id", "repo_path", "created_at", "window_days",
                    "total_commits", "hotspot_file_count", "dx_score"]
        return [dict(zip(columns, row)) for row in result]

    def _query_child(self, table: str, run_id: str) -> list[dict]:
        """Generic helper to query a child table by run_id."""
        result = self._conn.execute(
            f"SELECT * FROM {table} WHERE run_id = ?", [run_id]  # noqa: S608
        ).fetchall()
        if not result:
            return []
        cols = [desc[0] for desc in self._conn.description]
        return [dict(zip(cols, row)) for row in result]

    def get_hotspot_files(self, run_id: str) -> list[dict]:
        return self._query_child("hotspot_files", run_id)

    def get_knowledge_files(self, run_id: str) -> list[dict]:
        return self._query_child("knowledge_files", run_id)

    def get_coupling_pairs(self, run_id: str) -> list[dict]:
        return self._query_child("coupling_pairs", run_id)

    def get_file_pain(self, run_id: str) -> list[dict]:
        return self._query_child("file_pain", run_id)

    def get_anemia_classes(self, run_id: str) -> list[dict]:
        return self._query_child("anemia_classes", run_id)

    def get_complexity_functions(self, run_id: str) -> list[dict]:
        return self._query_child("complexity_functions", run_id)

    def get_cluster_summaries(self, run_id: str) -> list[dict]:
        return self._query_child("cluster_summaries", run_id)

    def get_cluster_drift(self, run_id: str) -> list[dict]:
        return self._query_child("cluster_drift", run_id)

    def get_effort_files(self, run_id: str) -> list[dict]:
        return self._query_child("effort_files", run_id)

    def get_dx_cognitive_files(self, run_id: str) -> list[dict]:
        return self._query_child("dx_cognitive_files", run_id)

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()
