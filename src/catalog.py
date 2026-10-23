"""The verified card catalog: load it and filter, sort and rank it exactly with pandas."""
import pandas as pd

from config import ROOT, load_config

SORTABLE = ["annual_fee_principal_rm", "annual_fee_supplementary_rm", "overseas_fee_pct", "best_rate_pct",
            "highest_rate_pct", "late_payment_cap_rm", "min_annual_income_rm"]
CARD_FIELDS = ["card_name", "bank", "card_type", "shariah_concept", "network",
               "annual_fee_principal_rm", "annual_fee_supplementary_rm", "annual_fee_note",
               "min_annual_income_rm", "late_payment_cap_rm", "overseas_fee_pct",
               "best_rate_pct", "highest_rate_pct", "rate_note", "benefits_text", "valid_as_at"]


class Catalog:
    def __init__(self, cfg=None):
        cfg = cfg or load_config()
        self.df = pd.read_csv(ROOT / cfg["catalog"]["path"], encoding="utf-8-sig")

    def query(self, bank=None, card_type=None, annual_income_rm=None, max_annual_fee_rm=None,
              name_contains=None, sort_by=None, limit=15):
        df = self.df
        if bank:
            df = df[df["bank"].str.contains(bank, case=False, na=False)]
        if card_type:
            df = df[df["card_type"].str.lower() == card_type.lower()]
        if name_contains:
            df = df[df["card_name"].str.contains(name_contains, case=False, na=False)]
        if annual_income_rm is not None:
            # Keep cards whose stated minimum is at or below the income, and cards whose PDS states none
            df = df[df["min_annual_income_rm"].isna() | (df["min_annual_income_rm"] <= annual_income_rm)]
        if max_annual_fee_rm is not None:
            df = df[df["annual_fee_principal_rm"] <= max_annual_fee_rm]
        if sort_by in SORTABLE:
            df = df.sort_values(sort_by, na_position="last")
        limit = max(1, min(int(limit or 15), 60))
        return len(df), df.head(limit)