"""Schema for one credit card in the catalog. Field descriptions guide the LLM extractor."""
from pydantic import BaseModel, Field


class Card(BaseModel):
    card_name: str = Field(description="Full card name as written, e.g. 'PB Quantum Visa/Mastercard Credit Card'")
    network: str | None = Field(None, description="Visa, Mastercard, Visa/Mastercard or American Express, if stated")
    annual_fee_principal_rm: float | None = Field(
        None, description="Principal card annual fee in RM. Use 0 if waived or free for life")
    annual_fee_supplementary_rm: float | None = Field(
        None, description="Supplementary card annual fee in RM. Use 0 if waived or free for life")
    annual_fee_note: str | None = Field(None, description="Conditions or notes on the annual fee, e.g. waiver rules")
    min_annual_income_rm: float | None = Field(None, description="Minimum annual income in RM, only if stated")
    late_payment_cap_rm: float | None = Field(
        None, description="Maximum late payment charge (or Ta'widh) in RM")
    late_payment_text: str | None = Field(None, description="Late payment charge wording from the document")
    overseas_fee_pct: float | None = Field(
        None, description="The bank's own charge on foreign currency transactions in percent, e.g. 1.25")
    overseas_fee_text: str | None = Field(None, description="Overseas transaction charge wording from the document")
    cash_advance_fee_text: str | None = Field(None, description="Cash advance or cash withdrawal fee wording")
    best_rate_pct: float | None = Field(
        None, description="Lowest finance or profit rate per annum (best payment-history tier)")
    highest_rate_pct: float | None = Field(None, description="Highest finance or profit rate per annum")
    benefits_text: str | None = Field(
        None, description="Benefits stated in this document only, e.g. travel takaful cover. Never add others")
    source_pages: list[int] = Field(default_factory=list, description="Page numbers where this card's facts appear")