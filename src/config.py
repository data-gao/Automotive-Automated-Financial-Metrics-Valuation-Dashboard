"""Configuration constants, paths, and peer group loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PEERS_CONFIG_PATH = DATA_DIR / "peers_config.json"

SUPPORTED_CURRENCIES = ("EUR", "USD", "CNY", "HKD")

# Fallback FX rates to EUR (used only if live FX fetch fails)
FALLBACK_FX_TO_EUR: Dict[str, float] = {
    "EUR": 1.0,
    "USD": 0.92,
    "CNY": 0.128,
    "HKD": 0.118,
}

# Yahoo Finance FX tickers quoted as 1 EUR in target currency or USD pairs
FX_YAHOO_PAIRS: Dict[str, str] = {
    "USD": "EURUSD=X",  # 1 EUR = X USD -> 1 USD = 1/X EUR
    "CNY": "EURCNY=X",  # 1 EUR = X CNY -> 1 CNY = 1/X EUR
    "HKD": "EURHKD=X",  # 1 EUR = X HKD -> 1 HKD = 1/X EUR
}


@dataclass(frozen=True)
class PeerConfig:
    """Configuration metadata for a single listed automotive peer."""

    ticker: str
    name: str
    region: str
    country: str
    trading_currency: str
    reporting_currency: str
    segment: str
    investment_angle: str
    category: str


@dataclass(frozen=True)
class DCFDefaults:
    """Default assumptions for the DCF valuation engine."""

    projection_years: int = 5
    risk_free_rate: float = 0.025  # 10Y Bund / blended EUR risk-free proxy
    equity_risk_premium: float = 0.055
    cost_of_debt_pre_tax: float = 0.045
    effective_tax_rate: float = 0.25
    terminal_growth_rate: float = 0.02
    exit_ebitda_multiple: float = 6.0


def load_peers_config(
    config_path: Path = PEERS_CONFIG_PATH,
    categories: Optional[List[str]] = None,
    include_disabled: bool = False,
) -> List[PeerConfig]:
    """Load peer configurations from JSON, optionally filtering by category."""
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    peers: List[PeerConfig] = []
    for cat_name, cat_data in raw.get("categories", {}).items():
        if categories and cat_name not in categories:
            continue
        if not include_disabled and not cat_data.get("enabled", True):
            continue
        for item in cat_data.get("peers", []):
            peers.append(
                PeerConfig(
                    ticker=item["ticker"],
                    name=item["name"],
                    region=item["region"],
                    country=item["country"],
                    trading_currency=item["trading_currency"],
                    reporting_currency=item["reporting_currency"],
                    segment=item["segment"],
                    investment_angle=item["investment_angle"],
                    category=cat_name,
                )
            )
    return peers


def get_available_categories(config_path: Path = PEERS_CONFIG_PATH) -> Dict[str, dict]:
    """Return metadata for all configured peer categories (e.g., OEM, Tier-1)."""
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("categories", {})
