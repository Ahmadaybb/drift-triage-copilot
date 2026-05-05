"""
schemas.py — Pydantic request/response models.
OWNER: Person 1
STATUS: ✅ DONE
"""
"""
schemas.py — Pydantic request/response models for the Bank Marketing model service.

Every field mirrors the UCI Bank Marketing dataset (bank-additional-full.csv).
`duration` is intentionally excluded — it leaks the target and is never accepted.
`pdays_never_contacted` is derived server-side from `pdays`, never sent by the client.
"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


# ── Categorical value sets ─────────────────────────────────────────────────────
# 'unknown' is a valid category everywhere it appears — it is informative,
# not missing data.

JobType = Literal[
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown",
]

MaritalStatus = Literal["divorced", "married", "single", "unknown"]

EducationLevel = Literal[
    "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
    "professional.course", "university.degree", "unknown",
]

DefaultStatus  = Literal["no", "yes", "unknown"]
HousingStatus  = Literal["no", "yes", "unknown"]
LoanStatus     = Literal["no", "yes", "unknown"]
ContactType    = Literal["cellular", "telephone"]

Month = Literal[
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]

DayOfWeek  = Literal["mon", "tue", "wed", "thu", "fri"]
POutcome   = Literal["failure", "nonexistent", "success"]


# ── Request model ──────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    """
    A single customer record for term deposit subscription prediction.
    All 19 features are required (duration is excluded — it leaks the target).
    """

    model_config = ConfigDict(extra="forbid")   # reject unknown fields

    # Numeric features
    age:              int   = Field(..., ge=17,  le=98,   description="Client age in years")
    campaign:         int   = Field(..., ge=1,   le=56,   description="Number of contacts during this campaign")
    pdays:            int   = Field(..., ge=0,   le=999,  description="Days since last contact (999 = never contacted)")
    previous:         int   = Field(..., ge=0,   le=7,    description="Number of contacts before this campaign")
    emp_var_rate:     float = Field(..., ge=-3.5, le=1.5, description="Employment variation rate (quarterly)")
    cons_price_idx:   float = Field(..., ge=92.0, le=95.0, description="Consumer price index (monthly)")
    cons_conf_idx:    float = Field(..., ge=-51.0, le=-26.0, description="Consumer confidence index (monthly)")
    euribor3m:        float = Field(..., ge=0.6,  le=5.1,  description="Euribor 3-month rate (daily)")
    nr_employed:      float = Field(..., ge=4963.6, le=5228.1, description="Number of employees (quarterly)")

    # Categorical features
    job:          JobType        = Field(..., description="Client job type")
    marital:      MaritalStatus  = Field(..., description="Marital status")
    education:    EducationLevel = Field(..., description="Education level")
    default:      DefaultStatus  = Field(..., description="Has credit in default?")
    housing:      HousingStatus  = Field(..., description="Has housing loan?")
    loan:         LoanStatus     = Field(..., description="Has personal loan?")
    contact:      ContactType    = Field(..., description="Contact communication type")
    month:        Month          = Field(..., description="Last contact month of year")
    day_of_week:  DayOfWeek      = Field(..., description="Last contact day of week")
    poutcome:     POutcome       = Field(..., description="Outcome of previous marketing campaign")

    def to_dataframe_row(self) -> dict:
        """
        Convert to a dict ready for the sklearn pipeline.
        - Renames fields back to dataset column names (emp_var_rate → emp.var.rate etc.)
        - Derives pdays_never_contacted from pdays server-side.
        """
        d = self.model_dump()

        # Rename fields that had dots replaced with underscores
        d["emp.var.rate"]    = d.pop("emp_var_rate")
        d["cons.price.idx"]  = d.pop("cons_price_idx")
        d["cons.conf.idx"]   = d.pop("cons_conf_idx")
        d["nr.employed"]     = d.pop("nr_employed")
        d["day_of_week"]     = d.pop("day_of_week")

        # Derive sentinel feature server-side — never accepted from client
        d["pdays_never_contacted"] = int(d["pdays"] == 999)

        return d


# ── Response models ────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    """Successful prediction response."""

    model_uri:   str   = Field(..., description="MLflow URI of the model that made this prediction")
    threshold:   float = Field(..., description="Operating threshold used for this prediction")
    probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability of subscription")
    label:       int   = Field(..., ge=0,   le=1,   description="Binary prediction (1=subscribe, 0=no)")


class ErrorDetail(BaseModel):
    """Single field-level validation error."""
    field:   str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Human-readable error message")


class ErrorResponse(BaseModel):
    """Structured error response — never a raw stack trace."""
    error:   str                  = Field(..., description="Error type")
    details: list[ErrorDetail]    = Field(default_factory=list)


# ── Drift webhook models ───────────────────────────────────────────────────────
class DriftFeatureReport(BaseModel):
    """Per-feature drift scores."""
    psi_scores:  dict[str, float] = Field(default_factory=dict, description="PSI scores for numeric features")
    chi2_pvals:  dict[str, float] = Field(default_factory=dict, description="Chi² p-values for categorical features")
    output_drift: float           = Field(..., ge=0.0, description="Output distribution drift score")


class DriftSeverity(BaseModel):
    """Overall drift severity classification."""
    level:   Literal["none", "warning", "high"] = Field(..., description="Drift severity level")
    reason:  str                                 = Field(..., description="Human-readable reason for severity")


class DriftAlert(BaseModel):
    """
    Webhook payload sent from model service to agent on every drift report change.
    This is the contract between the platform and the agent — treat schema changes as breaking.
    """
    event:       Literal["drift_alert"] = "drift_alert"
    model_uri:   str                    = Field(..., description="MLflow URI of the model being monitored")
    severity:    DriftSeverity
    drift_report: DriftFeatureReport
    timestamp:   str                    = Field(..., description="ISO 8601 UTC timestamp")


# ── Promotion models ───────────────────────────────────────────────────────────
class PromotionRequest(BaseModel):
    """
    Sent from the agent to the platform's /promote endpoint after human approval.
    This is the contract between the agent and the platform — treat schema changes as breaking.
    """
    action:           Literal["promote", "rollback"]
    model_name:       str   = Field(..., description="Registered model name in MLflow")
    version:          str   = Field(..., description="Model version to promote or roll back to")
    approved_by:      Literal["human"]
    investigation_id: str   = Field(..., description="Agent investigation ID that triggered this action")
    timestamp:        str   = Field(..., description="ISO 8601 UTC timestamp")


class PromotionResponse(BaseModel):
    """Response from the platform's /promote endpoint."""
    success:     bool
    model_uri:   str  = Field(..., description="New MLflow URI after promotion")
    message:     str