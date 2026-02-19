from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class RunSummary(BaseModel):
    run_id: str
    repo_path: str
    created_at: datetime
    window_days: int
    total_commits: int
    hotspot_file_count: int
    dx_score: float


class RunDetail(BaseModel):
    run_id: str
    repo_path: str
    created_at: datetime
    window_days: int
    from_date: datetime
    to_date: datetime
    total_commits: int
    first_commit_date: datetime | None
    last_commit_date: datetime | None
    hotspot_file_count: int
    developer_risk_index: float
    knowledge_island_count: int
    coupling_pair_count: int
    anemic_total_classes: int
    anemic_anemic_count: int
    anemic_anemic_pct: float
    anemic_average_ams: float
    complexity_total_functions: int
    complexity_avg: float
    complexity_max: int
    complexity_high_count: int
    clustering_k: int
    clustering_silhouette: float
    effort_total_files: int
    effort_model_r_squared: float
    effort_coefficients: list[float]
    dx_score: float
    dx_throughput: float
    dx_feedback_delay: float
    dx_focus_ratio: float
    dx_cognitive_load: float
    dx_weights: list[float]
    god_class_total_classes: int = 0
    god_class_god_count: int = 0
    god_class_god_pct: float = 0.0
    god_class_average_gcs: float = 0.0


class HotspotFile(BaseModel):
    run_id: str
    file_path: str
    change_frequency: int
    code_churn: int
    hotspot_score: float
    rework_ratio: float
    file_size: int


class KnowledgeFile(BaseModel):
    run_id: str
    file_path: str
    knowledge_concentration: float
    primary_author: str
    primary_author_pct: float
    is_knowledge_island: bool
    author_count: int


class CouplingPairRow(BaseModel):
    run_id: str
    file_a: str
    file_b: str
    shared_commits: int
    coupling_strength: float
    support: float
    expected_cochange: float
    lift: float


class FilePainRow(BaseModel):
    run_id: str
    file_path: str
    size_normalized: float
    volatility_normalized: float
    distance_normalized: float
    pain_score: float


class AnemicClassRow(BaseModel):
    run_id: str
    file_path: str
    class_name: str
    field_count: int
    behavior_method_count: int
    dbsi: float
    ams: float


class ComplexityFnRow(BaseModel):
    run_id: str
    file_path: str
    function_name: str
    line_number: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    max_nesting_depth: int
    length: int


class ClusterRow(BaseModel):
    run_id: str
    cluster_id: int
    label: str
    size: int
    centroid_file_count: float
    centroid_total_churn: float
    centroid_add_ratio: float


class DriftRow(BaseModel):
    run_id: str
    cluster_label: str
    first_half_pct: float
    second_half_pct: float
    drift: float
    trend: str


class EffortFileRow(BaseModel):
    run_id: str
    file_path: str
    rei_score: float
    proxy_label: float


class CognitiveRow(BaseModel):
    run_id: str
    file_path: str
    complexity_score: float
    coordination_score: float
    knowledge_score: float
    change_rate_score: float
    composite_load: float


class GodClassRow(BaseModel):
    run_id: str
    file_path: str
    class_name: str
    method_count: int
    field_count: int
    total_complexity: int
    cohesion: float
    god_class_score: float


class RunComparison(BaseModel):
    run_a: RunDetail
    run_b: RunDetail
    deltas: dict[str, float]
