**Rhythm Step Engine — MVP Contract**

---

## 1. Purpose

The Step Engine is responsible for generating **playable, musically coherent step charts** from audio input.

It is the **core product** of the system.
All UI, platforms, and future AI features are downstream of this engine.

---

## 2. Core Responsibilities

The Step Engine must:

* Analyze audio to extract rhythmic structure
* Construct a discrete musical timing grid
* Place steps aligned to that grid
* Select arrows using playability-aware logic
* Output a deterministic, inspectable chart format
* Allow holding steps (press a step and hold for a duration)

The engine must **never** place steps arbitrarily or rely on nondeterministic behavior.

---

## 3. Design Principles

### 3.1 Grid-Locked Timing (Non-Negotiable)

* All steps must align to a musical timing grid derived from BPM.
* Arbitrary timestamps are forbidden.
* Timing resolution is limited by difficulty.

---

### 3.2 Determinism

Given:

* The same audio input
* The same difficulty

The engine must produce **identical charts every time**.

A deterministic random seed is derived from:

* Audio hash
* Difficulty level

---

### 3.3 Algorithmic First

* Step placement and arrow selection are handled by deterministic logic.
* AI systems (if added later) may influence **high-level style only**:

  * Density shaping
  * Pattern preferences
  * Section emphasis

AI must never:

* Place individual arrows
* Decide exact timestamps

---

### 3.4 Playability Over Density

A chart that is slightly sparse but readable is always preferred over one that is dense but awkward.

---

## 4. Audio Analysis Contract

The Step Engine consumes the following derived audio features:

* **BPM** — global tempo estimate
* **Beat Times** — quarter-note positions
* **Onset Times** — percussive transients
* **Energy Curve** — RMS or equivalent
* **Duration** — total track length

## 5. Timing Grid

The timing grid is constructed from BPM.

Allowed subdivisions by difficulty:

| Difficulty | Grid Subdivision       |
| ---------- | ---------------------- |
| Easy       | Quarter notes (1/4)    |
| Medium     | Eighth notes (1/8)     |
| Hard       | Sixteenth notes (1/16) |

Steps may only be placed at grid positions.

---

## 6. Step Placement Logic

For each grid position:

1. Check for an onset within ±30ms
2. If an onset is present:

   * The grid position becomes a strong candidate
3. If no onset is present:

   * Placement depends on difficulty density limits

Density constraints are enforced per second to prevent over-charting.

---

## 7. Arrow Selection & Foot Logic

The engine maintains state across steps:

* Last arrow used
* Last foot used (left / right)

Arrow selection is governed by a **cost function**.

### Penalized Patterns

* Repeating the same arrow consecutively
* Excessive same-foot usage
* Crossovers (Easy / Medium)

### Preferred Patterns

* Alternating feet
* Lateral movement
* Simple directional flow

At each step, the arrow (or arrow pair) with the **lowest total cost** is selected.

---

## 8. Jumps (Two-Arrow Steps)

Jumps are allowed only if:

* Difficulty ≥ Medium
* The step occurs on a downbeat
* Energy exceeds a defined threshold

Constraints:

* No consecutive jumps
* Jumps must be physically plausible (two-foot)

---

## 9. Difficulty Profiles

Difficulty is defined by **density and complexity**, not raw speed.

```json
{
  "easy": {
    "max_steps_per_second": 1.0,
    "allow_jumps": false,
    "grid": "quarter"
  },
  "medium": {
    "max_steps_per_second": 2.0,
    "allow_jumps": true,
    "grid": "eighth"
  },
  "hard": {
    "max_steps_per_second": 3.5,
    "allow_jumps": true,
    "grid": "sixteenth"
  }
}
```

---

## 10. Output Contract

The Step Engine outputs a chart in JSON format:

```json
{
  "bpm": 128,
  "offset": 0.42,
  "difficulty": "medium",
  "steps": [
    { "time": 1.875, "arrows": ["left"] },
    { "time": 2.250, "arrows": ["up"] },
    { "time": 2.625, "arrows": ["down", "right"] }
  ]
}
```

All step times must correspond exactly to timing grid positions.

---

## 11. Non-Goals (MVP)

The Step Engine explicitly does **not** handle:

* Scoring or judgments
* Visual presentation
* Player input handling
* Calibration or latency compensation
* Genre or key detection
* Advanced note types (holds, mines, etc.)

---

## 12. Success Criteria

The Step Engine is considered successful if:

* Charts feel rhythmically aligned to the music
* Easy charts are playable by non-experts
* Hard charts are dense but readable
* Output is deterministic and reproducible
* Charts can be debugged visually with a simple renderer

---

**If the step engine feels good, the game works. Everything else is optional.**
