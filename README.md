# Customer Retention: From Prediction to Product Decision

### Prioritizing limited outreach capacity—and testing whether ML earns its place.

**Python · scikit-learn · Churn Modeling · Product Strategy · Experiment Design**

A retention team selects customers for proactive support using a business rule, and it can only make a limited number of contacts. Should it replace the rule with an ML model?

Using 7,043 telecom customer records, I compared ML and rule-based targeting under a fixed outreach budget, then designed a randomized experiment to test whether ML selects a better outreach list than a strong rule, with a business-as-usual group to measure the incremental impact of each outreach policy.

| 🎯 Who to target | ⚖️ ML vs. rules (offline) | 🧪 Experiment |
|---|---|---|
| **Month-to-month customers**: 54% of customers, but **88% of churners** | Logistic regression identifies **287 churners** in a 423-customer shortlist (68% precision vs. 43% random) | **Primary question:** does ML outreach retain more customers than rule outreach? |
| Focus on month-to-month customers for a near-term retention experiment | A strong multi-factor rule identifies **270**: ML adds **+17 (+6%)**. A tenure-only rule reaches 230 | **Supporting question:** does either outreach policy improve retention over business-as-usual? |

*Status: Portfolio case study on the public IBM Telco Customer Churn dataset. Offline analysis complete; experiment proposed, not run.*

---

## 1 · The Product Decision

**Decision:** Should the retention team replace its rule-based targeting with an ML model?

The offline benchmark shows whether ML identifies more observed churners under the same contact budget. The experiment tests whether that screening advantage translates into more customers retained.

**ML vs. Rule** answers the replacement decision. Comparisons with **business-as-usual** establish whether either outreach policy creates incremental value.

## 2 · Who Is Eligible: Month-to-Month Customers

The initial experiment focuses on month-to-month customers to align targeting with a near-term cancellation outcome. This group contains most observed churners in the test set.

| Test set (2,113 customers) | Share of customers | Share of churners | Churn rate |
|---|---:|---:|---:|
| Month-to-month | 54% | **88%** | 43% |
| One- or two-year contract | 46% | 12% | — |

**Eligibility:** month-to-month customers only. In practice, customers whose contracts end within the window would also qualify; the dataset has no contract dates, so they are not modeled here.

**Capacity:** For this case study, the assumed outreach budget is equivalent to 20% of the full customer base, which is **423 contacts** in the test set. All selected customers come from the month-to-month pool (about 37% of that pool). The 423-customer figure is the offline list size, not the experiment sample size.

## 3 · Offline Results: ML vs. Rules Within the Eligible Pool

The model is trained on all customers, then used to rank month-to-month customers only. I compared it with two rules a retention team could run without a model:

- **Tenure rule:** shortest tenure first.
- **Multi-factor rule:** one point each for tenure ≤ 12 months, fiber-optic internet, and electronic-check payment; ties go to shorter tenure. Each condition was checked on training data only (churn 52% vs. 33%, 55% vs. 28%, and 53% vs. 33%).

<img width="700" height="450" alt="different_reachout_capacity" src="https://github.com/user-attachments/assets/e8134788-dd31-44e5-977f-b3a624e52764" />

Observed churners identified (precision in parentheses), with all customers selected from the month-to-month pool:

| Customers selected (% of full base) | Logistic regression | Boosting (GBM) | Multi-factor rule | Tenure rule |
|---|---:|---:|---:|---:|
| 106 (5%) | 85 (80%) | 93 (88%) | 76 (72%) | 64 (60%) |
| 212 (10%) | 158 (75%) | 162 (76%) | 151 (71%) | 129 (61%) |
| **423 (20%)** | **287 (68%)** | **287 (68%)** | **270 (64%)** | **230 (54%)** |
| 634 (30%) | 370 (58%) | 375 (59%) | 365 (58%) | 317 (50%) |

*Random selection within month-to-month customers: 43% precision. The tenure rule breaks ties randomly; across 1,000 tie-breaks its 423-contact result ranges from 225 to 233.*

### What the results show

- **Logistic regression over boosting.** Both identify 287 churners at 423 selected customers. Boosting shows its largest advantage at smaller lists, so I recommend the simpler, more interpretable LR model at this operating point.
- **A good rule gets most of the way.** The multi-factor rule identifies 94% as many churners as ML. The two lists overlap by 77%. Where they differ, ML's picks are more often true churners: **61% (59 of 96) vs. 44% (42 of 96)**.
- **ML identifies 17 additional churners in the test sample:** +6% at the 423-customer list size over the strong rule, and +57 over the tenure-only rule. A tenure-only benchmark would overstate ML's value. The strong rule is the right comparison for the experiment.
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

**Why support first:** Test whether resolving service friction improves retention without introducing a pricing change. Both outreach arms use the same support protocol, so the policy difference is *who* is selected. Discounts can be explored separately.

## 5 · Experiment Design

### Arms

Randomly assign eligible month-to-month customers to three arms. Each outreach arm selects the same share of its assigned population (about 37%) and uses the same contact-attempt limits, channels, and support scope.

| Arm | Allocation (example) | Who gets a check-in |
|---|---:|---|
| **ML outreach** | 45% | Highest LR risk scores (model frozen before launch) |
| **Rule outreach** | 45% | Highest multi-factor rule scores |
| **Holdout (business-as-usual)** | 10% | No additional outreach |

**Proposed allocation: 45% ML / 45% Rule / 10% business-as-usual.** Prioritize sample for the main policy comparison; finalize allocation through power analysis for both the primary and supporting comparisons. Freeze eligibility, model, rule, tie-breaking, and delivery procedures before launch.

### Metrics, by role

| Role | Metric | What it answers |
|---|---|---|
| **Primary** | Churn rate in ML arm − churn rate in rule arm | **Does ML select a better list?** This is the decision metric |
| **Incremental impact** | Churn in each outreach arm vs. holdout | Does either outreach policy improve retention over existing operations? |
| **Business** | Net value difference: retained contribution − support and model costs, per assigned customer | Is "better" worth the cost of running a model? |
| **Confirmation** | 90-day retention, same comparisons | Does the retention benefit persist beyond the primary window? |
| **Guardrails** | Complaints, marketing opt-outs, support workload | Does either policy cause harm? |
| **Diagnostics** | Contact rate, support uptake, issue resolution | Why did the results come out this way? |

**Churn definition:** a cancellation request within two billing cycles (~60 days) of assignment. Cancellation requests avoid the billing-cycle lag in "active subscription" status.

**Whole-arm comparison (intention-to-treat):** Every assigned customer counts, contacted or not. ML and the rule select different people, so comparing only contacted customers would mix selection with effect. Comparing whole randomized arms isolates the policy.

### Sample Size and Timing

Power the primary ML–Rule comparison using the historical 60-day cancellation rate and the smallest whole-arm improvement that would justify the model's additional cost. Use **two-sided α = 0.05** and **80% power**, and verify that the holdout is sufficiently sized for the supporting comparisons. Predefine the testing approach across comparisons.

Set enrollment duration from the required sample size and eligible traffic. Complete every customer's 60-day observation window before the primary readout, followed by the 90-day assessment.

### Rollout decision

| Evidence | Decision |
|---|---|
| ML improves retention over Rule and business-as-usual; additional value covers model costs; guardrails pass | **Adopt ML targeting** |
| Rule improves retention over business-as-usual; the estimated ML advantage is too small to justify its cost | **Keep the rule** |
| Both policies fall short of the required business benefit, with sufficiently precise estimates | **Revise the intervention or delivery** |
| Evidence remains inconclusive | **Keep the current approach and plan additional evidence collection** |

Use 90-day retention and contribution to assess the duration of the benefit and update the rollout value calculation.

## 6 · Limits and Next Steps

- **Single snapshot.** One public dataset with no timestamps or intervention history. Performance on future cohorts and real save rates are unknown.
- **Assumed capacity.** The 20% budget reflects human support. ML matters most when contacts are scarce and costly. With near-zero-cost automated outreach, the question shifts from *who to contact* to *whether contacting helps or annoys*.
- **Prediction ≠ persuasion.** The model ranks who is likely to leave, not who will respond. Once experiment data exists, **uplift modeling** can target the customers most likely to be saved.
- **Before launch:** validate on time-split cohorts, check calibration, and agree on baselines, costs, and logging with the support team.

---

## Explore the Work

[**Python workflow →**](https://github.com/ysun-data/Churn-Analysis-new/blob/main/churn_analysis.py) · [**Data & repository →**](https://github.com/ysun-data/Churn-Analysis-new) · [**Original R analysis →**](https://github.com/ysun-data/Telecom-Churn-Analysis)

```bash
pip install pandas numpy scikit-learn statsmodels shap matplotlib
python churn_analysis.py
```
