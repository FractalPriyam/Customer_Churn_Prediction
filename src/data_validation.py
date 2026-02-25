"""Data validation model for the Telco Customer Churn dataset.

This module defines `ChurnValidation`, a Pydantic model that enforces
expected data types and simple constraints for each column used by the
churn prediction pipeline. Fields are grouped by logical sections:
- demographic info
- account info
- target variable (Churn)

The model also includes a validator to handle empty or whitespace-only
strings in the `TotalCharges` column which is sometimes represented as
an empty string in CSV exports.
"""

from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Literal, Optional


class ChurnValidation(BaseModel):
    """Pydantic model describing expected fields and simple constraints.

    Notes:
    - Use `Literal` for categorical fields to validate enumerated values.
    - Numeric fields have basic bounds where appropriate (e.g., `tenure`).
    - `TotalCharges` may be missing or blank in raw CSVs, handled below.
    """

    # -----------------------------
    # Demographic information
    # -----------------------------
    # Unique customer identifier (string)
    customerID: str

    # Gender of the customer: must be 'Male' or 'Female'
    gender: Literal["Male", "Female"]

    # Senior citizen flag: 0 (not senior) or 1 (senior)
    SeniorCitizen: Literal[0, 1]

    # Partner and dependents flags: 'Yes' or 'No'
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    # Internet service type
    InternetService: Literal['DSL', 'Fiber optic', 'No']

    # Options for services that may be 'No', 'Yes', or 'No internet service'
    OnlineSecurity: Literal['No', 'Yes', 'No internet service']
    OnlineBackup: Literal['Yes', 'No', 'No internet service']
    TechSupport: Literal['No', 'Yes', 'No internet service']
    StreamingTV: Literal['No', 'Yes', 'No internet service']
    StreamingMovies: Literal['No', 'Yes', 'No internet service']

    # -----------------------------
    # Account information
    # -----------------------------
    # Number of months the customer has been with the company
    tenure: int = Field(ge=0)

    # Contract type: month-to-month, one year, or two year
    Contract: Literal["Month-to-month", "One year", "Two year"]

    # Whether billing is paperless
    PaperlessBilling: Literal["Yes", "No"]

    # Accepted payment methods
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    # Recurring monthly charge; must be positive
    MonthlyCharges: float = Field(gt=0)

    # Total charges may be missing or blank in some raw datasets.
    TotalCharges: Optional[float] = None

    # -----------------------------
    # Target variable
    # -----------------------------
    # Whether the customer churned: 'Yes' or 'No'
    Churn: Optional[Literal["Yes", "No"]] = None

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def handle_empty_total_charges(cls, v):
        """Coerce empty or whitespace-only `TotalCharges` strings to `None`.

        Some CSV exports represent missing numeric values as empty strings.
        This validator runs before type coercion and turns those empty
        strings into `None` so downstream logic can decide how to handle
        missing totals (e.g., impute with 0 or fill based on `tenure`).
        """
        if isinstance(v, str) and v.strip() == "":
            return None
        return v