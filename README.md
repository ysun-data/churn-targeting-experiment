# Customer Retention: From Prediction to Product Decision

### Prioritizing limited outreach capacity—and testing whether ML earns its place.

**Python · scikit-learn · Churn Modeling · Product Strategy · Experiment Design**

A retention team selects customers for proactive support using a business rule, and it can only make a limited number of contacts. Should it replace the rule with an ML model?

Using 7,043 telecom customer records, I compared ML and rule-based targeting under a fixed outreach budget, then designed a randomized experiment to test whether ML selects a better outreach list than a strong rule, with a holdout group to confirm that outreach itself works.

| 🎯 Who to target | ⚖️ ML vs. rules (offline) | 🧪 Experiment |
|---|---|---|
| **Month-to-month customers**: 54% of customers, but **88% of churners** | Logistic regression reaches **287 churners** with 423 contacts (68% precision vs. 43% random) | **Primary question:** does ML outreach retain more customers than rule outreach? |
| Long-contract customers rarely churn in the test window, so they're excluded | A strong multi-factor rule reaches **270**: ML adds **+17 (+6%)**. A tenure-only rule reaches 230 | **Holdout check:** does outreach beat business-as-usual, so the comparison is meaningful? |

*Status: Portfolio case study on the public IBM Telco Customer Churn dataset. Offline analysis complete; experiment proposed, not run.*

---

## 1 · The Product Decision

**Decision:** Should the retention team replace its rule-based targeting with an ML model?

Customers saved by an outreach program depend on two things:

> **customers saved = at-risk customers on the list × share that outreach saves**

- **The list** is what ML could improve. Offline data can estimate it.
- **The save rate** depends on the intervention. Only an experiment can measure it.

So the experiment compares ML against the rule directly, and keeps a small no-outreach holdout. If outreach saves no one, ML and the rule will look identical, but only because neither works. The holdout separates "ML is no better" from "outreach doesn't work."

## 2 · Who Is Eligible: Month-to-Month Customers

Customers on one- or two-year contracts face early-termination fees and rarely churn within an experiment window. Contacting them uses capacity without changing outcomes.

| Test set (2,113 customers) | Share of customers | Share of churners | Churn rate |
|---|---:|---:|---:|
| Month-to-month | 54% | **88%** | 43% |
| One- or two-year contract | 46% | 12% | — |

**Eligibility:** month-to-month customers only. In practice, customers whose contracts end within the window would also qualify; the dataset has no contract dates, so they are not modeled here.

**Capacity:** The outreach budget is fixed by support staffing at the equivalent of 20% of the full customer base, which is **423 contacts** in the test set. All contacts are drawn from month-to-month customers (about 37% of that pool).

## 3 · Offline Results: ML vs. Rules Within the Eligible Pool

The model is trained on all customers, then used to rank month-to-month customers only. I compared it with two rules a retention team could run without a model:

- **Tenure rule:** shortest tenure first.
- **Multi-factor rule:** one point each for tenure ≤ 12 months, fiber-optic internet, and electronic-check payment; ties go to shorter tenure. Each condition was checked on training data only (churn 52% vs. 33%, 55% vs. 28%, and 53% vs. 33%).

<img width="700" height="450" alt="different_reachout_capacity" src="https://github.com/user-attachments/assets/e8134788-dd31-44e5-977f-b3a624e52764" />

Churners reached (precision in parentheses), all contacts drawn from month-to-month customers:

| Contacts (% of full base) | Logistic regression | Boosting (GBM) | Multi-factor rule | Tenure rule |
|---|---:|---:|---:|---:|
| 106 (5%) | 85 (80%) | 93 (88%) | 76 (72%) | 64 (60%) |
| 212 (10%) | 158 (75%) | 162 (76%) | 151 (71%) | 129 (61%) |
| **423 (20%)** | **287 (68%)** | **287 (68%)** | **270 (64%)** | **230 (54%)** |
| 634 (30%) | 370 (58%) | 375 (59%) | 365 (58%) | 317 (50%) |

*Random selection within month-to-month customers: 43% precision. The tenure rule breaks ties randomly; across 1,000 tie-breaks its 423-contact result ranges from 225 to 233.*

### What the results show

- **Logistic regression over boosting.** Both reach 287 churners at 423 contacts. Boosting only pulls ahead at very small lists, so the simpler, more interpretable model is the better choice at this operating point.
- **A good rule gets most of the way.** The multi-factor rule reaches 94% of ML's churners. The two lists overlap by 77%. Where they differ, ML's picks are more often true churners: **61% (59 of 96) vs. 44% (42 of 96)**.
- **ML's edge is real but small:** +17 churners per 423 contacts (+6%) over the strong rule, and +57 over the tenure-only rule. A tenure-only benchmark would overstate ML's value. The strong rule is the right comparison for the experiment.
- **Early tenure matters most.** It is the strongest single signal, which points to onboarding and service friction as the place to intervene.

<details>
<summary><b>Technical depth behind the decision</b> (click to expand)</summary>

<br>

| Technical decision | What I did | Why it matters |
|---|---|---|
| Keep new users in the pipeline | Traced 11 blank billing values to zero-tenure customers and represented them as not yet billed | Avoids excluding customers entering a high-risk stage |
| Reduce duplicated information | Verified overlapping service labels row by row and consolidated them while retaining parent service fields | Preserves customer context with fewer redundant inputs |
| Validate feature choices | Tested four feature sets across LR, random forest, and GBM on shared cross-validation splits | Removing billing fields preserved LR/GBM performance but hurt RF; feature decisions depend on the model |
| Evaluate where the experiment runs | Measured Precision@N within month-to-month customers at fixed contact counts | Matches the offline evidence to the experiment's population and capacity |
| Build a fair benchmark | Derived the multi-factor rule from training data and checked tie-breaking robustness | Avoids comparing ML against a deliberately weak rule |

Stratified 10-fold cross-validation and a held-out 30% test set. Test ROC-AUC: 0.844 (LR) and 0.848 (GBM) overall; 0.753 and 0.762 within month-to-month customers.

</details>

## 4 · Intervention: Proactive Support Check-ins

**Hypothesis:** Some early churn comes from unresolved setup or service issues that proactive support can fix.

**Customer experience:** Selected customers receive a short call or message asking whether their service is working as expected. Reported issues are routed to support.

**Why support, not discounts:** Discounts subsidize customers who would have stayed anyway, and they mix price effects into the test. Both outreach arms get the identical support protocol, so the only difference between them is *who* is contacted.

## 5 · Experiment Design

### Arms

Randomly assign eligible month-to-month customers to three arms. Each outreach arm contacts the same share of its arm (about 37%).

| Arm | Allocation (example) | Who gets a check-in |
|---|---:|---|
| **ML outreach** | 45% | Highest LR risk scores (model frozen before launch) |
| **Rule outreach** | 45% | Highest multi-factor rule scores |
| **Holdout (business-as-usual)** | 10% | No additional outreach |

The holdout can be small because outreach vs. no outreach is a large difference. Most of the sample goes to the ML and rule arms, where the difference to detect is small.

### Metrics, by role

| Role | Metric | What it answers |
|---|---|---|
| **Primary** | Churn rate in ML arm − churn rate in rule arm | **Does ML select a better list?** This is the decision metric |
| **Precondition check** | Churn in each outreach arm vs. holdout | Does outreach work? If not, the primary comparison can't be interpreted |
| **Business** | Net value difference: retained contribution − support and model costs, per assigned customer | Is "better" worth the cost of running a model? |
| **Confirmation** | 90-day retention, same comparisons | Did outreach prevent churn, or only delay it a billing cycle? |
| **Guardrails** | Complaints, marketing opt-outs, support workload | Does either policy cause harm? |
| **Diagnostics** | Contact rate, support uptake, issue resolution | Why did the results come out this way? |

**Churn definition:** a cancellation request within two billing cycles (~60 days) of assignment. Cancellation requests avoid the billing-cycle lag in "active subscription" status.

**Whole-arm comparison (intention-to-treat):** Every assigned customer counts, contacted or not. ML and the rule select different people, so comparing only contacted customers would mix selection with effect. Comparing whole randomized arms isolates the policy.

### Sizing the primary comparison

**Expected effect:** the difference between arms is roughly

> share contacted × precision gap × save rate ≈ 37% × 4 points × save rate

That is about **1/17 of the outreach-vs.-holdout effect**, so the ML–rule test needs far more customers (roughly 300×) than testing whether outreach works. It is a realistic test for a large operator, and cheaper with low-cost digital outreach.

**Minimum detectable effect (MDE):** set from the business case. It is the smallest ML–rule retention gap that would cover the cost of running and maintaining the model. Size the test at two-sided α = 0.05 and 80% power for that gap.

**Break-even illustration:** Per 10,000 contacts, ML reaches about **400 more at-risk customers** than the rule (17 per 423). Its extra value is:

> 400 × save rate × value per saved customer − model cost

For an operator making 100,000 contacts a month, with $200 per saved customer and $15k a month to run the model, ML breaks even if outreach saves about **2%** of the at-risk customers it reaches. If available traffic can't detect a gap that small, the save rate measured in the holdout comparison can feed this calculation as a fallback.

### Rollout decision

| Outcome | Decision |
|---|---|
| Outreach doesn't beat the holdout | **Rethink the intervention.** ML vs. rule can't be judged when neither list is acted on effectively |
| Outreach works; ML beats the rule; net value positive; guardrails pass | **Adopt ML targeting** |
| Outreach works; ML–rule gap is precisely estimated below the MDE | **Keep the rule.** The model doesn't pay for itself |
| Outreach works; ML–rule result inconclusive | **Extend the test** or decide with the break-even check using the measured save rate |
| Lift fades by 90 days | **Treat it as delay, not retention**; redesign before scaling |

## 6 · Limits and Next Steps

- **Single snapshot.** One public dataset with no timestamps or intervention history. Performance on future cohorts and real save rates are unknown.
- **Assumed capacity.** The 20% budget reflects human support. ML matters most when contacts are scarce and costly. With near-zero-cost automated outreach, the question shifts from *who to contact* to *whether contacting helps or annoys*.
- **Prediction ≠ persuasion.** The model ranks who is likely to leave, not who will respond. Once experiment data exists, **uplift modeling** can target the customers most likely to be saved.
- **Before launch:** validate on time-split cohorts, check calibration, and agree on baselines, costs, and logging with the support team.

---

## Explore the Work

<!-- TODO: update links after renaming repositories -->
[**Python workflow →**](https://github.com/ysun-data/Churn-Analysis-new/blob/main/churn_analysis.py) · [**Data & repository →**](https://github.com/ysun-data/Churn-Analysis-new) · [**Original R analysis →**](https://github.com/ysun-data/Telecom-Churn-Analysis)

```bash
pip install pandas numpy scikit-learn statsmodels shap matplotlib
python churn_analysis.py
```
