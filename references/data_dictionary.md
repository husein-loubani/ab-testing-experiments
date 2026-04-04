# Data Dictionary

## Fast Food Marketing Campaign

**Source:** IBM Watson Analytics sample dataset (`WA_Marketing-Campaign.csv`)

| Variable | Type | Description |
|---|---|---|
| `MarketID` | Categorical | Market identifier |
| `MarketSize` | Categorical | Small, Medium, or Large |
| `LocationID` | Categorical | Store identifier |
| `AgeOfStore` | Numeric | Store age in years |
| `Promotion` | Categorical (1, 2, 3) | Promotion variant assigned to the store |
| `week` | Numeric (1-4) | Week of the test period |
| `SalesInThousands` | Numeric | Weekly sales in thousands of dollars |

**Design:** Each store was assigned to exactly one promotion and observed for four weeks.

---

## Cookie Cats

**Source:** Kaggle, Cookie Cats A/B test dataset (`cookie_cats.csv`)

| Variable | Type | Description |
|---|---|---|
| `userid` | Integer | Unique player identifier |
| `version` | Categorical | `gate_30` (control) or `gate_40` (treatment) |
| `sum_gamerounds` | Numeric | Total game rounds played in the first 14 days |
| `retention_1` | Boolean (0/1) | Whether the player returned on Day 1 |
| `retention_7` | Boolean (0/1) | Whether the player returned on Day 7 |

**Design:** Players were randomly assigned at install to either gate_30 (first gate at level 30) or gate_40 (first gate at level 40). The gate is a forced wait that encourages in-app purchases.
