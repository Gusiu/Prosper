"""Configuration management using Pydantic Settings."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from prosper.domain import DEPTH_BIN_LABELS

_UNSET = Path()


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_prefix="PROSPER_",
        case_sensitive=False,
        extra="ignore",
    )

    # Data paths (a single root that can be overridden by CLI `--root`).
    # The four sub-directories are derived from `data_root` unless explicitly
    # set. They are non-optional so callers never have to narrow away a `None`
    # that cannot occur once construction has finished.
    data_root: Path = Field(default=Path("./data"), description="Root directory for all data")
    raw_data_dir: Path = _UNSET
    processed_data_dir: Path = _UNSET
    reports_dir: Path = _UNSET
    meta_dir: Path = _UNSET

    @model_validator(mode="after")
    def _derive_and_create_directories(self) -> "Settings":
        defaults = {
            "raw_data_dir": self.data_root / "raw",
            "processed_data_dir": self.data_root / "processed",
            "reports_dir": self.data_root / "reports",
            "meta_dir": self.data_root / "meta",
        }
        for name, default in defaults.items():
            if getattr(self, name) == _UNSET:
                object.__setattr__(self, name, default)

        for directory in (self.data_root, *(getattr(self, name) for name in defaults)):
            directory.mkdir(parents=True, exist_ok=True)
        return self

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

    # Label settings
    label_depth_bins_default: list[str] = Field(default_factory=lambda: list(DEPTH_BIN_LABELS))

    # Baseline model settings. The window must stay wider than the longest
    # horizon (365d) plus a usable sample count, otherwise the long horizon has
    # no realised outcomes to count; see prosper.predict.window.
    baseline_rolling_window_days: int = Field(default=730, ge=1)
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
    # The round trip a signal has to pay for before it is worth acting on.
    # This is a property of the market and the venue, not of the forecast, so
    # it belongs here rather than in the labels: moving markets is a config
    # change, not a relabel-and-retrain. 0.4% is 2x(fee + slippage) for Binance
    # SPOT taker orders; venues with wider spreads want a larger value.
    planner_round_trip_cost: float = Field(
        default=0.004,
        ge=0.0,
        le=1.0,
        description="Round-trip cost a signal must clear before the planner acts",
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
