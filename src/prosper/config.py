"""Configuration management using Pydantic Settings."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_prefix="PROSPER_",
        case_sensitive=False,
        extra="ignore",
    )

    # Data paths (a single root that can be overridden by CLI `--root`)
    data_root: Path = Field(default=Path("./data"), description="Root directory for all data")
    raw_data_dir: Path | None = None
    processed_data_dir: Path | None = None
    reports_dir: Path | None = None
    meta_dir: Path | None = None

    # Binance API
    binance_public_data_base: str = "https://data.binance.vision"
    binance_api_base: str = "https://api.binance.com"

    # Download settings
    download_workers: int = Field(default=4, ge=1, le=16)
    download_timeout: int = Field(default=300, ge=10)
    download_retry_max: int = Field(default=3, ge=1)
    download_retry_backoff: float = Field(default=2.0, ge=1.0)

    # API rate limiting
    api_rate_limit_requests_per_minute: int = Field(default=1200, ge=1)
    api_rate_limit_backoff_base: float = Field(default=2.0, ge=1.0)

    # QA settings
    qa_gap_tolerance_seconds: int = Field(default=120, ge=0)
    qa_flat_threshold_default: float = Field(default=0.01, ge=0.0, le=1.0)

    # Label settings
    label_flat_threshold_default: float = Field(default=0.01, ge=0.0, le=1.0)
    label_depth_bins_default: list[str] = Field(
        default_factory=lambda: ["1-2", "2-3", "3-5", "5-8", "8-13", "13-21", "21-34", "34+"]
    )

    # Baseline model settings
    baseline_rolling_window_days: int = Field(default=180, ge=1)
    baseline_min_samples: int = Field(default=30, ge=1)

    # Planner settings
    planner_short_weeks: tuple[int, int] = Field(
        default=(1, 26), description="Short horizon weeks range"
    )
    planner_medium_weeks: tuple[int, int] = Field(
        default=(13, 52), description="Medium horizon weeks range"
    )
    planner_long_weeks: tuple[int, int] = Field(
        default=(26, 104), description="Long horizon weeks range"
    )

    # Trading simulation costs (for eval walk-forward MVP)
    trading_fee_rate: float = Field(
        default=0.001, ge=0.0, le=1.0, description="Fee rate per position"
    )
    trading_slippage_proxy_rate: float = Field(
        default=0.001, ge=0.0, le=1.0, description="Slippage proxy rate per position"
    )

    # Research / reproducibility settings
    strict: bool = Field(
        default=False, description="Fail fast on data quality issues (research mode)"
    )
    deterministic: bool = Field(
        default=False, description="Enable deterministic training (research mode)"
    )
    save_metadata: bool = Field(
        default=False, description="Save meta.json alongside artifacts (research mode)"
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility (research mode)"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )

    def __init__(self, config_file: Path | None = None, **kwargs: Any) -> None:
        """Initialize settings, optionally loading from YAML config file."""
        if config_file and config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f) or {}
            kwargs = {**yaml_config, **kwargs}

        super().__init__(**kwargs)

        # Derive directory layout from data_root (so CLI can isolate smoke runs).
        if self.raw_data_dir is None:
            self.raw_data_dir = self.data_root / "raw"
        if self.processed_data_dir is None:
            self.processed_data_dir = self.data_root / "processed"
        if self.reports_dir is None:
            self.reports_dir = self.data_root / "reports"
        if self.meta_dir is None:
            self.meta_dir = self.data_root / "meta"

        # Ensure directories exist
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    @property
    def raw_binance_spot_klines_1m_dir(self) -> Path:
        """Directory for raw Binance SPOT 1m klines ZIP files."""
        return self.raw_data_dir / "binance" / "spot" / "klines" / "1m"

    @property
    def processed_binance_spot_klines_dir(self) -> Path:
        """Directory for processed Binance SPOT klines Parquet files."""
        return self.processed_data_dir / "binance" / "spot" / "klines"

    @property
    def processed_binance_spot_labels_dir(self) -> Path:
        """Directory for processed Binance SPOT labels Parquet files."""
        return self.processed_data_dir / "binance" / "spot" / "labels"

    @property
    def reports_qa_dir(self) -> Path:
        """Directory for QA reports."""
        return self.reports_dir / "qa"

    @property
    def reports_predictions_dir(self) -> Path:
        """Directory for prediction reports."""
        return self.reports_dir / "predictions"

    @property
    def reports_recommendations_dir(self) -> Path:
        """Directory for recommendation reports."""
        return self.reports_dir / "recommendations"

    @property
    def reports_eval_dir(self) -> Path:
        """Directory for evaluation reports."""
        return self.reports_dir / "eval"


# Global settings instance
_settings: Settings | None = None


def get_settings(
    config_file: Path | None = None, data_root: Path | None = None, **kwargs: Any
) -> Settings:
    """
    Get settings.

    If `data_root` or any research kwargs are provided, returns a fresh Settings instance
    (so CLI smoke runs can isolate their data lake and research flags take effect).
    """
    global _settings
    if data_root is not None or kwargs:
        return Settings(config_file=config_file, data_root=data_root, **kwargs)
    if _settings is None:
        _settings = Settings(config_file=config_file)
    return _settings
