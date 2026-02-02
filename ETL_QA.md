# ETL Quality Assurance Checklist
**Objective:** 1:1 Reproduction of PandemicLLM Data Architecture (Section 8.1)

## 1. Scale & Resolution
- [ ] **Volume:** Total samples must exceed 50,000.
    - *Status:* PASS (54k achieved via sliding window).
- [ ] **Resolution:** Daily Sliding Windows (Shift = 1 day).
    - *Status:* PASS (Implemented in loop).
- [ ] **Interpolation:** Must use **Cubic/Linear** smoothing, NOT step-functions.
    - *Status:* PASS (`interp1d` linear implemented).
- [ ] **Jitter:** Must include Gaussian noise to prevent overfitting on smooth lines.
    - *Status:* PASS (5% noise injection active).

## 2. Multimodal Streams (The "Context")
- [ ] **Hospitalizations:** Primary signal (Target).
    - *Status:* PASS (NHSN via Delphi).
- [ ] **Lead Indicator:** Must be a *leading* signal (Cases/ED Visits).
    - *Status:* PASS (CDC ED Visits `7mra-9cq9`).
- [ ] **Vaccination:** Must separate Dose 1, Series Complete, Boosters.
    - *Status:* **FAIL**. Code currently only maps `series_complete`. Paper used 3 distinct curves.
- [ ] **Variants:** Must map Transmission, Escape, Severity.
    - *Status:* PASS (Mapping table `BIO_PROFILES` active).

## 3. Temporal Integrity (The "Zipper")
- [ ] **Synchronization:** Join key must be a **Date Object**, not a string/week-index.
    - *Status:* **FAIL**. Recent crash showed `Timestamp` vs `str` mismatch.
- [ ] **Lead Offset:** Lead indicator must be shifted by -7 days (Cause -> Effect).
    - *Status:* PASS (Implied in loop logic `lookback_idx`).
- [ ] **Deduplication:** Variants/ED data must be unique per day.
    - *Status:* **PARTIAL**. Patch applied but needs verification.

## 4. Sociological DNA
- [ ] **Static Features:** Full 25-feature vector from legacy pickle.
    - *Status:* PASS (Loader function active).

## 5. Failure Mode
- [ ] **Assertions:** Script must CRASH on default values (0.4 / 0.78).
    - *Status:* **FAIL**. Fallback logic (`except: vec = BIO_PROFILES['JN']`) is still present. This allows silent data corruption.

---

### **Remediation Plan:**
1.  **Add Vax Layers:** Fetch and map `administered_dose1_pop_pct` and `additional_doses_vax_pct` alongside `series_complete`.
2.  **Fix Timestamp:** Ensure `week_ending` is explicitly cast to `pd.Timestamp` at ingestion.
3.  **Kill Defaults:** Remove the `try-except` blocks around lookup logic. If a state/date is missing context, **drop the sample**. Do not guess.
