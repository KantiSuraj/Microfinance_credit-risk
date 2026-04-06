# Bug Fix Implementation Plan v2 — Microfinance RL Environment
> Supersedes v1. All three decisions from v1 reversed or refined per critical review.

---

## Decision Log (What Changed from v1 and Why)

| # | v1 Decision | v2 Decision | Reason |
|---|-------------|-------------|--------|
| Bug 3 | Option A: return `MonitoringObservation` with `month_number=0, observed_status=ON_TIME` | **Option B**: keep `ApplicantObservation`, separate internal state from exposed observation | Option A injects fabricated `ON_TIME` payment signal at month 0 → corrupts RL training |
| Bug 2 | Track `collectedDocs` as client-side JS object | **Derive entirely from `obs.documents_submitted`** | Dual source of truth breaks on page refresh, async race, direct API calls |
| Bug 1 | Enable buttons after `resetEnv()` synchronously | **Enable buttons inside `.then()` after fetch resolves** | Race condition: buttons could fire on a not-yet-started episode |
| Bug 3 UI | `const sq = obs.signal_quality ?? 0` | **Explicit null check, show `—` or `Transitioning...`** | Missing value ≠ zero value; `0%` is misleading |
| Bug 3 backend | `obs.current_phase = "APPLICATION"` mutated after `_obs_phase1()` | **Pass `current_phase` explicitly into `_obs_phase1()`** | If `_obs_phase1` reads from `state.phase` internally, the mutation gets stomped before serialization |
| Bug 3 transition | UI has no signal it's in a transition step | **Add `"transitioning_to_phase2": true` field to APPROVE response** | UI can render "Phase 2 starting…" intentionally rather than appearing delayed or broken |
| Bug 1 button reset | `enableAllActionButtons()` blindly enables all buttons regardless of phase | **Derive which buttons to enable from `obs.current_phase`** | If reset somehow lands in MONITORING phase, Phase 1 buttons must not be re-enabled |

---

## Bug 1 — Buttons Unclickable After Second "Start Episode"

### Root Cause
`doStep()` sets `button.disabled = true` on all `.action-btn` elements before each step. Its `finally` block re-enables only buttons for the **current phase**. When an episode ends (`episodeActive = false`), those buttons stay permanently disabled. `resetEnv()` never re-enables them, so the next episode starts with all buttons frozen.

### Fix — Frontend only (`index.html`)

Re-enable buttons **inside the fetch `.then()` callback**, not after it. This prevents a race where buttons are live before the backend confirms the new episode has started.

```js
// ❌ WRONG — synchronous, race condition possible
function resetEnv() {
    fetch('/reset').then(res => res.json()).then(obs => updateObservation(obs));
    document.querySelectorAll('.action-btn').forEach(b => b.disabled = false); // too early
}

// ✅ CORRECT — enable only after backend confirms reset
function resetEnv() {
    fetch('/reset')
        .then(res => res.json())
        .then(obs => {
            updateObservation(obs);                    // renders new applicant profile
            enableButtonsForPhase(obs.current_phase);  // phase-aware, not blind enable
        });
}

// ❌ WRONG — blindly enables all buttons regardless of phase
// function enableAllActionButtons() {
//     document.querySelectorAll('.action-btn').forEach(b => b.disabled = false);
// }

// ✅ CORRECT — phase-aware: only enable buttons that belong to the current phase
function enableButtonsForPhase(phase) {
    // Always clear collected markers on all buttons (safe for both phases)
    document.querySelectorAll('.action-btn').forEach(b => b.classList.remove('collected'));

    if (phase === 'APPLICATION' || phase == null) {
        // Enable Phase 1 buttons, disable Phase 2 buttons
        document.querySelectorAll('#phase1Actions .action-btn').forEach(b => b.disabled = false);
        document.querySelectorAll('#phase2Actions .action-btn').forEach(b => b.disabled = true);
        document.getElementById('phase1Info').classList.remove('hidden');
        document.getElementById('phase2Info').classList.add('hidden');
        document.getElementById('phase1Actions').classList.remove('hidden');
        document.getElementById('phase2Actions').classList.add('hidden');
    } else if (phase === 'MONITORING') {
        // Enable Phase 2 buttons, disable Phase 1 buttons
        document.querySelectorAll('#phase1Actions .action-btn').forEach(b => b.disabled = true);
        document.querySelectorAll('#phase2Actions .action-btn').forEach(b => b.disabled = false);
        document.getElementById('phase1Info').classList.add('hidden');
        document.getElementById('phase2Info').classList.remove('hidden');
        document.getElementById('phase1Actions').classList.add('hidden');
        document.getElementById('phase2Actions').classList.remove('hidden');
    }
}
```

### Why This Is Sufficient
The `doStep()` `finally` block logic does not need to change. The only missing piece was re-enabling on reset — now it's gated behind a confirmed backend response.

---

## Bug 2 — Redundant Doc Requests Keep Decreasing Reward (No Visual Feedback)

### Root Cause
The backend **correctly** penalizes redundant requests (escalating penalty, not flat). The problem is entirely UI — buttons look identical whether or not a document has been collected, so users (and judges evaluating RL agents) have no idea they're taking a penalized action.

### Pre-condition: Verify Backend Returns `documents_submitted`

Before implementing the UI fix, confirm that the Phase 1 observation schema includes a `documents_submitted` field. Check the `_obs_phase1()` return value in `microfinance_env_environment.py`:

```python
# If this field is missing, add it explicitly:
return ApplicantObservation(
    ...
    documents_submitted=list(state.docs_collected),  # ADD THIS if absent
    ...
)
```

The field name in the list must match exactly what the UI checks. Canonical names to use:
- `"income_proof"` — for REQUEST_INCOME_PROOF
- `"credit_history"` — for REQUEST_CREDIT_HISTORY  
- `"senior_review"` — for FLAG_FOR_REVIEW

### Fix — Frontend only (`index.html`)

**Never track `collectedDocs` as a separate JS object.** Derive the collected state entirely from `obs.documents_submitted` on every `updateObservation()` call. The UI becomes stateless w.r.t. collection — it just reflects backend truth.

```js
// ✅ CORRECT — derive, never track
function updateObservation(obs) {
    // ... existing rendering logic ...

    // Sync collected-doc visual state from backend
    const docs = obs.documents_submitted || [];
    document.getElementById('btnIncome').classList.toggle('collected', docs.includes('income_proof'));
    document.getElementById('btnCredit').classList.toggle('collected', docs.includes('credit_history'));
    document.getElementById('btnReview').classList.toggle('collected', docs.includes('senior_review'));
}
```

This also handles reset automatically: when a new episode starts, `obs.documents_submitted` is `[]`, so all `.collected` classes are removed without needing a separate reset call.

```css
/* CSS for collected state */
.action-btn.collected {
    border: 1px solid #22c55e;     /* green border */
    opacity: 0.75;
    position: relative;
}
.action-btn.collected::after {
    content: '✓';
    position: absolute;
    top: 4px;
    right: 6px;
    color: #22c55e;
    font-size: 11px;
}
```

### UX Decision: Keep Buttons Clickable (Not Disabled)
Do **not** fully disable collected-doc buttons. This is an RL environment — the agent must be able to receive the penalty feedback to learn from it. Disabling the buttons removes a valid (if costly) action from the action space. The `.collected` visual style is a human hint only.

---

## Bug 3 — Signal Quality = NaN on Phase Transition

### Root Cause
When APPROVE is clicked:
1. Backend internally sets `state.phase = Phase.MONITORING`
2. But returns `_obs_phase1(...)` with `current_phase = "MONITORING"` in the serialized response
3. UI sees `current_phase === "MONITORING"` → enters Phase 2 render path
4. Tries to read `obs.signal_quality` → `undefined` (not in Phase 1 schema) → `undefined * 100 = NaN`
5. Same for `obs.month_number` → `"undefined / NaN"`

### Fix — Option B (Conservative, RL-Correct)

**Internal state and exposed observation must be separated.** The backend transitions internally, but the observation returned for the APPROVE step stays as Phase 1. Phase 2 observation begins on the *next* step.

#### Backend (`microfinance_env_environment.py`)

**Critical constraint:** `_obs_phase1()` must NOT re-read `state.phase` internally to set `current_phase`. If it does, any override you apply after the fact gets stomped before serialization. The phase must be passed in explicitly as a parameter.

```python
# ❌ WRONG — _obs_phase1 re-derives phase from state internally
def _obs_phase1(self, msg, done, reward):
    return ApplicantObservation(
        current_phase=self.state.phase.value,  # reads MONITORING → override is lost
        ...
    )

# ✅ CORRECT — accept explicit current_phase parameter
def _obs_phase1(self, msg, done, reward, current_phase=None):
    return ApplicantObservation(
        current_phase=(current_phase or self.state.phase.value),
        ...
    )
```

Then in the APPROVE handler:

```python
elif atype == "APPROVE":
    # ... existing reward calculation (unchanged) ...
    s.phase = Phase.MONITORING          # internal state transitions ✅

    # Pass current_phase explicitly — do NOT let _obs_phase1 read state.phase
    return self._obs_phase1(
        msg=f"APPROVED at {conf:.0%} confidence. Phase 2 starts next step — "
            f"signal quality {s.signal_quality:.0%}. Phase 1 reward: {reward:+.3f}.",
        done=False,
        reward=reward,
        current_phase="APPLICATION",         # explicit override ✅
        transitioning_to_phase2=True,        # UI transition signal ✅
    )
```

`transitioning_to_phase2` must also be added to the `ApplicantObservation` schema (default `False`):

```python
@dataclass
class ApplicantObservation:
    ...
    transitioning_to_phase2: bool = False   # ADD: signals APPROVE transition step
```

The next call to `step()` calls `_obs_phase2()` because `state.phase` is already `MONITORING`. That observation will have `transitioning_to_phase2 = False` (the default), so the UI knows the transition is complete.

#### Frontend (`index.html`) — Transition-aware rendering + defensive null guards

The UI now has an explicit signal to work with. The `transitioning_to_phase2` flag means the UI does not need to infer a "limbo" state from missing fields — it knows intentionally.

```js
function updateObservation(obs) {
    const phase = obs.current_phase;
    const isTransitioning = obs.transitioning_to_phase2 === true;

    // Phase-aware button enable on every observation update, not just reset
    enableButtonsForPhase(phase);

    if (phase === 'APPLICATION') {
        if (isTransitioning) {
            // APPROVE step: still showing Phase 1 layout, but signal the upcoming transition
            document.getElementById('phaseStatusBanner').textContent = 'Phase 2 starting next step…';
            document.getElementById('phaseStatusBanner').classList.remove('hidden');
        } else {
            document.getElementById('phaseStatusBanner').classList.add('hidden');
        }
        // ... existing Phase 1 field rendering (unchanged) ...
    }

    if (phase === 'MONITORING') {
        document.getElementById('phaseStatusBanner').classList.add('hidden');

        const sq = obs.signal_quality;
        const monthNum = obs.month_number;
        const monthsRem = obs.months_remaining;
        const totalM = (monthNum ?? 0) + (monthsRem ?? 0);

        // ✅ null check — missing ≠ zero
        document.getElementById('vSignalQ').textContent =
            (sq != null && !isNaN(sq)) ? `${(sq * 100).toFixed(0)}%` : '—';

        document.getElementById('vMonth').textContent =
            (monthNum != null) ? `${monthNum} / ${totalM}` : 'Transitioning...';

        document.getElementById('vObservedPayment').textContent = obs.observed_payment ?? '—';
        document.getElementById('vCumulativeMisses').textContent = obs.cumulative_misses ?? '—';
    }

    // Bug 2: derive collected-doc state from backend, never track client-side
    const docs = obs.documents_submitted || [];
    document.getElementById('btnIncome').classList.toggle('collected', docs.includes('income_proof'));
    document.getElementById('btnCredit').classList.toggle('collected', docs.includes('credit_history'));
    document.getElementById('btnReview').classList.toggle('collected', docs.includes('senior_review'));
}
```

> **Key architectural point:** `enableButtonsForPhase(phase)` is now called inside `updateObservation()` on every render — not just on reset. This means the button state is always derived from the observation, never from a separate event. Reset, step, and page-load all flow through the same path.

---

## Verification Plan

### Pre-implementation Checks
- [ ] Confirm `obs.documents_submitted` exists in Phase 1 API response (print raw JSON after a doc request step)
- [ ] Confirm exact string values for doc names (`"income_proof"` etc.) match between backend and UI `includes()` checks
- [ ] Confirm `_obs_phase1()` does **not** re-read `state.phase` internally — if it does, the `current_phase` parameter fix is mandatory before anything else
- [ ] Confirm `ApplicantObservation` dataclass can accept new `transitioning_to_phase2: bool = False` field without breaking existing serialization

### Automated Tests
```bash
python -m pytest test_env.py -v   # baseline — must pass before and after
```

Add these specific cases to `test_env.py`:
```python
def test_approve_returns_application_phase_with_transition_flag():
    """APPROVE response must have current_phase=APPLICATION and transitioning_to_phase2=True."""
    obs = env.step("APPROVE")
    assert obs["current_phase"] == "APPLICATION"          # not MONITORING
    assert obs["transitioning_to_phase2"] is True         # explicit transition signal
    assert obs.get("signal_quality") is None              # Phase 1 schema, no Phase 2 fields

def test_second_step_after_approve_returns_clean_phase2():
    """Step after APPROVE must return full MonitoringObservation, no transition flag."""
    env.step("APPROVE")
    obs = env.step("DO_NOTHING")
    assert obs["current_phase"] == "MONITORING"
    assert obs.get("transitioning_to_phase2") is not True  # flag cleared
    assert obs["signal_quality"] is not None
    assert 0.0 <= obs["signal_quality"] <= 1.0
    assert obs["month_number"] is not None

def test_redundant_doc_request_still_returns_documents_submitted():
    """documents_submitted must be present in obs even after redundant request."""
    env.step("REQUEST_INCOME_PROOF")
    obs = env.step("REQUEST_INCOME_PROOF")
    assert "documents_submitted" in obs
    assert "income_proof" in obs["documents_submitted"]

def test_reset_returns_application_phase():
    """Reset must always return current_phase=APPLICATION so buttons are correctly initialized."""
    obs = env.reset()
    assert obs["current_phase"] == "APPLICATION"
    assert obs.get("transitioning_to_phase2") is not True
```

### Manual Browser Verification

| Scenario | Expected Result |
|----------|----------------|
| Complete episode → Start Episode → click any Phase 1 button | Button responds, step executes |
| Start Episode → check Phase 2 buttons | Phase 2 buttons disabled (not just hidden) |
| Request Income Proof → inspect button | Green border + ✓ visible, button still clickable |
| Request Income Proof again | Penalty in trajectory log, ✓ still showing |
| Approve loan → check Phase 1 panel | Phase 1 still visible, banner shows "Phase 2 starting next step…" |
| Approve loan → check Phase 1 action buttons | Phase 1 buttons disabled (transitioning), Phase 2 buttons still disabled |
| Approve loan → click DO_NOTHING | Phase 2 panel appears, Signal Quality shows real %, never NaN |
| Start new episode after Phase 2 | All Phase 1 buttons enabled, Phase 2 buttons disabled, no ✓ markers |

---

## Summary of All File Changes

| File | Change | Bug |
|------|--------|-----|
| `index.html` | Move button enable inside `.then()` callback; replace with `enableButtonsForPhase(obs.current_phase)` | Bug 1 |
| `index.html` | Call `enableButtonsForPhase(phase)` inside `updateObservation()` on every render — not just on reset | Bug 1 |
| `index.html` | Remove `collectedDocs` JS object; derive from `obs.documents_submitted` via `.toggle('collected', ...)` | Bug 2 |
| `index.html` | Add `.collected` CSS class + toggle logic for income, credit, review buttons | Bug 2 |
| `index.html` | Replace `?? 0` guards with explicit null checks showing `—`/`Transitioning...` | Bug 3 |
| `index.html` | Add `phaseStatusBanner` element; show "Phase 2 starting next step…" when `obs.transitioning_to_phase2 === true` | Bug 3 |
| `microfinance_env_environment.py` | Add `current_phase` parameter to `_obs_phase1()` — must not re-read `state.phase` internally | Bug 3 |
| `microfinance_env_environment.py` | APPROVE passes `current_phase="APPLICATION"` and `transitioning_to_phase2=True` explicitly | Bug 3 |
| `microfinance_env_environment.py` | Add `transitioning_to_phase2: bool = False` to `ApplicantObservation` dataclass | Bug 3 |
| `microfinance_env_environment.py` | Ensure `documents_submitted` field is serialized in Phase 1 observation | Bug 2 pre-condition |
