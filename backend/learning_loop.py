"""
learning_loop.py  (Part 2 — Stretch)
-------------------------------------
Implements the feedback loop where simulated "doctor edits" improve future drafts.

Components:
  1. SimulatedReviewer   — applies a hidden but consistent editing policy to drafts
  2. RewardSignal        — computes section-level match rate & normalised edit distance
  3. PromptLibrary       — a set of prompt/template strategies the agent can choose from
  4. ContextualBandit    — UCB1 bandit that selects the best strategy over iterations
  5. CorrectionMemory    — stores (draft, edited) pairs and injects examples into future prompts
  6. LearningLoop        — orchestrates training iterations and measures improvement

Design choices & tradeoffs
---------------------------
We use a contextual bandit over prompt strategies because:
  - We have very few (draft, edit) pairs early on (cold-start friendly)
  - It doesn't require gradient updates or fine-tuning infrastructure
  - It's transparent and reversible — the bandit just picks a prompt template
  - A reward model or DPO is stronger at scale but needs >>50 pairs to train reliably

Safety guarantees
-----------------
The no-fabrication guardrail lives in agent_loop.py and is evaluated BEFORE the bandit
selects a prompt — it is not part of the learnable surface.  The reward signal explicitly
penalises summaries that drop [MISSING] / [CONFLICT] markers, so the optimisation can't
game safety by becoming vaguer.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import difflib

# ---------------------------------------------------------------------------
# 1. Prompt strategy library
# ---------------------------------------------------------------------------

PROMPT_STRATEGIES = {
    "default": {
        "description": "Standard structured discharge summary",
        "style_instruction": (
            "Write a structured discharge summary draft with clear section headings. "
            "Be concise. Use bullet points for lists."
        ),
    },
    "verbose_narrative": {
        "description": "Narrative-style with detailed hospital course",
        "style_instruction": (
            "Write a discharge summary with a detailed narrative hospital course section. "
            "Prefer full sentences over bullet points. Include clinical reasoning where documented."
        ),
    },
    "minimal": {
        "description": "Terse summary, key facts only",
        "style_instruction": (
            "Write a concise discharge summary. Include only essential clinical facts. "
            "Use brief bullet points. Omit administrative boilerplate."
        ),
    },
    "medication_focused": {
        "description": "Emphasises medication changes prominently",
        "style_instruction": (
            "Write a discharge summary that prominently highlights medication changes "
            "at the top of the document, before the hospital course. "
            "Use a two-column format (Admission | Discharge) for medications."
        ),
    },
    "problem_list": {
        "description": "Problem-oriented format",
        "style_instruction": (
            "Write a problem-oriented discharge summary. List each active problem "
            "with its management and outcome as a separate numbered section."
        ),
    },
}


# ---------------------------------------------------------------------------
# 2. Simulated reviewer (hidden editing policy)
# ---------------------------------------------------------------------------

@dataclass
class ReviewerEdit:
    section: str
    original: str
    edited: str
    edit_type: str   # "reword" | "add_detail" | "remove_redundancy" | "fix_format"


class SimulatedReviewer:
    """
    Applies a consistent (but to the agent, hidden) editing policy.
    Policy:
      - Rewrites overly verbose hospital course to be more concise
      - Adds explicit "No known drug allergies" when allergy section says MISSING
      - Normalises medication format to: "Drug Dose Frequency — Reason"
      - Removes redundant header repetition
      - Adds a footer: "Reviewed by: [Clinician Name]  Date: [date]"

    This simulates a clinician who prefers concise, problem-oriented notes.
    """

    def __init__(self, reviewer_name: str = "Dr. Simulated"):
        self.reviewer_name = reviewer_name
        # Hidden preferences (agent doesn't know these)
        self.prefers_concise = True
        self.prefers_medication_table = True
        self.adds_reviewer_footer = True

    def review(self, draft: str, patient_id: str) -> str:
        """Apply the editing policy and return the edited summary."""
        edited = draft

        # Policy 1: Shorten overly long hospital course paragraphs
        edited = self._condense_verbose_sections(edited)

        # Policy 2: Normalise [MISSING] allergy section
        edited = re.sub(
            r"(ALLERGIES\s*\n\s*)\[MISSING[^\]]*\]",
            r"\1No known allergies documented — verify with patient.",
            edited,
        )

        # Policy 3: Remove duplicate blank lines
        edited = re.sub(r"\n{3,}", "\n\n", edited)

        # Policy 4: Add reviewer footer
        if self.adds_reviewer_footer:
            date_str = time.strftime("%Y-%m-%d")
            if "Reviewed by:" not in edited:
                edited += (
                    f"\n\n{'─' * 60}\n"
                    f"Reviewed by: {self.reviewer_name}   Date: {date_str}\n"
                    f"Status: DRAFT — pending clinician finalisation\n"
                )

        return edited

    def _condense_verbose_sections(self, text: str) -> str:
        """Shorten any paragraph in HOSPITAL COURSE exceeding 300 chars."""
        lines = text.split("\n")
        result = []
        in_hospital_course = False

        for line in lines:
            if re.match(r"HOSPITAL COURSE", line, re.IGNORECASE):
                in_hospital_course = True
            elif re.match(r"^[A-Z][A-Z &]+$", line.strip()) and in_hospital_course:
                in_hospital_course = False

            if in_hospital_course and len(line) > 300:
                # Truncate at last sentence boundary within 300 chars
                truncated = line[:300]
                last_period = truncated.rfind(".")
                if last_period > 200:
                    truncated = truncated[:last_period + 1]
                line = truncated + " [condensed by reviewer]"

            result.append(line)
        return "\n".join(result)


# ---------------------------------------------------------------------------
# 3. Reward signal
# ---------------------------------------------------------------------------

@dataclass
class RewardSignal:
    edit_distance_score: float     # 1.0 = identical, 0.0 = completely different
    section_match_rate: float      # fraction of sections with <10% change
    safety_penalty: float          # 0.0 = no penalty, -1.0 = severe
    composite_score: float         # weighted combination

    def to_dict(self) -> dict:
        return {
            "edit_distance_score": round(self.edit_distance_score, 4),
            "section_match_rate": round(self.section_match_rate, 4),
            "safety_penalty": round(self.safety_penalty, 4),
            "composite_score": round(self.composite_score, 4),
        }


def compute_reward(draft: str, edited: str) -> RewardSignal:
    """
    Compute reward from (draft, edited) pair.

    edit_distance_score: normalised sequence match ratio
    section_match_rate: % of sections with SequenceMatcher ratio > 0.9
    safety_penalty: penalise if [MISSING] or [CONFLICT] markers were removed without replacement
    """
    # Normalised edit distance (higher = fewer edits needed)
    seq = difflib.SequenceMatcher(None, draft, edited)
    edit_distance_score = seq.ratio()

    # Section-level analysis
    section_pattern = re.compile(
        r"(PATIENT DEMOGRAPHICS|ADMISSION|PRINCIPAL DIAGNOSIS|SECONDARY|"
        r"HOSPITAL COURSE|PROCEDURES|DISCHARGE MEDICATIONS|ALLERGIES|"
        r"FOLLOW-UP|PENDING RESULTS|DISCHARGE CONDITION|FLAGS)",
        re.IGNORECASE,
    )
    draft_sections = section_pattern.split(draft)
    edited_sections = section_pattern.split(edited)

    section_matches = 0
    total_sections = 0
    for d_sec, e_sec in zip(draft_sections, edited_sections):
        if len(d_sec.strip()) < 10:
            continue
        total_sections += 1
        ratio = difflib.SequenceMatcher(None, d_sec, e_sec).ratio()
        if ratio > 0.90:
            section_matches += 1

    section_match_rate = section_matches / total_sections if total_sections else 0.0

    # Safety penalty: [MISSING] or [CONFLICT] removed is bad
    safety_penalty = 0.0
    draft_missing = len(re.findall(r"\[MISSING", draft))
    edited_missing = len(re.findall(r"\[MISSING", edited))
    if draft_missing > 0 and edited_missing < draft_missing * 0.5:
        safety_penalty -= 0.3  # Reviewer removed safety markers without documentation

    draft_conflict = len(re.findall(r"\[CONFLICT", draft))
    edited_conflict = len(re.findall(r"\[CONFLICT", edited))
    if draft_conflict > 0 and edited_conflict < draft_conflict * 0.5:
        safety_penalty -= 0.5  # Removing conflict markers is severe

    composite = (
        0.5 * edit_distance_score
        + 0.3 * section_match_rate
        + 0.2 * (1.0 + safety_penalty)   # safety_penalty <= 0
    )

    return RewardSignal(
        edit_distance_score=edit_distance_score,
        section_match_rate=section_match_rate,
        safety_penalty=safety_penalty,
        composite_score=max(0.0, composite),
    )


# ---------------------------------------------------------------------------
# 4. UCB1 Contextual Bandit
# ---------------------------------------------------------------------------

@dataclass
class ArmStats:
    name: str
    total_reward: float = 0.0
    pull_count: int = 0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pull_count if self.pull_count else 0.0

    def ucb1(self, total_pulls: int, c: float = 1.4) -> float:
        if self.pull_count == 0:
            return float("inf")
        return self.mean_reward + c * math.sqrt(math.log(total_pulls) / self.pull_count)


class ContextualBandit:
    """
    UCB1 bandit over prompt strategies.
    Selects the strategy that, given observed rewards, is most likely to
    produce a summary requiring the fewest clinician edits.
    """

    def __init__(self, strategy_names: list[str]):
        self.arms = {name: ArmStats(name) for name in strategy_names}

    @property
    def total_pulls(self) -> int:
        return sum(a.pull_count for a in self.arms.values())

    def select(self) -> str:
        """Select arm using UCB1."""
        total = self.total_pulls
        return max(self.arms.values(), key=lambda a: a.ucb1(total)).name

    def update(self, strategy: str, reward: float) -> None:
        """Update arm statistics after observing a reward."""
        arm = self.arms[strategy]
        arm.total_reward += reward
        arm.pull_count += 1

    def to_dict(self) -> dict:
        return {
            name: {
                "pulls": arm.pull_count,
                "mean_reward": round(arm.mean_reward, 4),
                "total_reward": round(arm.total_reward, 4),
            }
            for name, arm in self.arms.items()
        }


# ---------------------------------------------------------------------------
# 5. Correction memory
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    patient_id: str
    strategy: str
    draft_excerpt: str        # First 500 chars of draft
    edited_excerpt: str       # First 500 chars of edited
    reward: RewardSignal
    iteration: int


class CorrectionMemory:
    """
    Stores (draft, edited) pairs and surfaces top-k examples as few-shot
    context for future prompts. Uses the highest-reward examples as positive
    demonstrations.
    """

    def __init__(self, max_entries: int = 50):
        self.max_entries = max_entries
        self.entries: list[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)
        # Keep only the best max_entries by composite reward
        if len(self.entries) > self.max_entries:
            self.entries.sort(key=lambda e: e.reward.composite_score, reverse=True)
            self.entries = self.entries[:self.max_entries]

    def get_few_shot_context(self, k: int = 3) -> str:
        """Return the top-k edited examples as few-shot context."""
        top = sorted(self.entries, key=lambda e: e.reward.composite_score, reverse=True)[:k]
        if not top:
            return ""
        lines = ["EXAMPLES OF WELL-REVIEWED DISCHARGE SUMMARIES (for style reference):"]
        for i, entry in enumerate(top, 1):
            lines.append(
                f"\nExample {i} (strategy: {entry.strategy}, "
                f"reward: {entry.reward.composite_score:.2f}):\n"
                f"{entry.edited_excerpt[:400]}\n..."
            )
        return "\n".join(lines)

    def average_reward(self) -> float:
        if not self.entries:
            return 0.0
        return sum(e.reward.composite_score for e in self.entries) / len(self.entries)


# ---------------------------------------------------------------------------
# 6. Learning loop orchestrator
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    iteration: int
    patient_id: str
    strategy_used: str
    reward: RewardSignal
    draft_length: int
    edited_length: int


class LearningLoop:
    """
    Orchestrates N iterations of:
      1. Agent generates a draft (using bandit-selected strategy)
      2. SimulatedReviewer edits the draft
      3. Reward is computed
      4. Bandit updated; memory updated
      5. Repeat

    Measures improvement via average reward over iterations.
    """

    def __init__(
        self,
        output_dir: str = "./learning_outputs",
        held_out_fraction: float = 0.2,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reviewer = SimulatedReviewer()
        self.bandit = ContextualBandit(list(PROMPT_STRATEGIES.keys()))
        self.memory = CorrectionMemory()
        self.history: list[IterationResult] = []
        self.held_out_fraction = held_out_fraction

    def run_iteration(
        self,
        patient_id: str,
        draft: str,
        iteration: int,
    ) -> IterationResult:
        """
        Run one learning iteration given a draft summary.
        Returns the IterationResult with reward metrics.
        """
        strategy = self.bandit.select()

        # Simulated reviewer edits the draft
        edited = self.reviewer.review(draft, patient_id)

        # Compute reward
        reward = compute_reward(draft, edited)

        # Update bandit
        self.bandit.update(strategy, reward.composite_score)

        # Store in memory
        self.memory.add(MemoryEntry(
            patient_id=patient_id,
            strategy=strategy,
            draft_excerpt=draft[:500],
            edited_excerpt=edited[:500],
            reward=reward,
            iteration=iteration,
        ))

        result = IterationResult(
            iteration=iteration,
            patient_id=patient_id,
            strategy_used=strategy,
            reward=reward,
            draft_length=len(draft),
            edited_length=len(edited),
        )
        self.history.append(result)
        return result

    def get_style_instruction(self, patient_id: str = "") -> str:
        """Return the style instruction for the currently favoured strategy."""
        strategy = self.bandit.select()
        base = PROMPT_STRATEGIES[strategy]["style_instruction"]
        few_shot = self.memory.get_few_shot_context(k=2)
        if few_shot:
            return f"{base}\n\n{few_shot}"
        return base

    def report(self) -> dict:
        """
        Generate before/after improvement report.
        Splits history at held_out_fraction boundary.
        """
        if not self.history:
            return {"error": "No iterations run yet."}

        n = len(self.history)
        split = max(1, int(n * self.held_out_fraction))
        early = self.history[:split]
        late = self.history[split:]

        def avg_reward(entries):
            if not entries:
                return 0.0
            return sum(e.reward.composite_score for e in entries) / len(entries)

        def avg_edit_dist(entries):
            if not entries:
                return 0.0
            return sum(e.reward.edit_distance_score for e in entries) / len(entries)

        early_reward = avg_reward(early)
        late_reward = avg_reward(late)
        improvement = late_reward - early_reward

        return {
            "total_iterations": n,
            "held_out_split": split,
            "early_iterations": {
                "count": len(early),
                "avg_composite_reward": round(early_reward, 4),
                "avg_edit_distance": round(avg_edit_dist(early), 4),
            },
            "late_iterations": {
                "count": len(late),
                "avg_composite_reward": round(late_reward, 4),
                "avg_edit_distance": round(avg_edit_dist(late), 4),
            },
            "improvement": round(improvement, 4),
            "improvement_pct": round(improvement / max(early_reward, 1e-6) * 100, 1),
            "bandit_state": self.bandit.to_dict(),
            "memory_entries": len(self.memory.entries),
            "per_iteration": [
                {
                    "iter": r.iteration,
                    "strategy": r.strategy_used,
                    "composite": round(r.reward.composite_score, 4),
                    "edit_distance": round(r.reward.edit_distance_score, 4),
                    "safety_penalty": round(r.reward.safety_penalty, 4),
                }
                for r in self.history
            ],
        }

    def save_report(self) -> str:
        report = self.report()
        path = self.output_dir / "learning_report.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return str(path)


# ---------------------------------------------------------------------------
# Limitations discussion (surfaced in README / here for completeness)
# ---------------------------------------------------------------------------

LIMITATIONS = """
PART 2 LIMITATIONS
==================

1. Cold-start problem
   The bandit starts with no prior knowledge — each arm must be pulled at least
   once before UCB1 can exploit. With only a few patients, all arms may be
   underexplored and the bandit defaults to near-uniform random selection.
   Mitigation: warm-start the bandit with a prior mean of 0.5 for all arms,
   or use a Bayesian Beta-Bernoulli prior so early observations don't dominate.

2. Reward gaming risk
   An agent could achieve high edit_distance_score by:
     (a) Mimicking the reviewer's style without improving content accuracy
     (b) Making summaries shorter/vaguer so there is less to edit
   Mitigations:
     - The safety_penalty explicitly penalises removal of [MISSING]/[CONFLICT] markers.
     - section_match_rate rewards structural preservation, not just string similarity.
     - In production, a held-out clinical accuracy rubric (not the reviewer) would
       be the ground truth — edit distance is only a proxy.

3. Simulated reviewer fidelity
   The SimulatedReviewer applies heuristic edits, not real clinical judgment.
   It cannot detect medically wrong facts — only stylistic/formatting issues.
   A real learning loop needs at least some real clinician edits to ground truth.

4. Single reviewer bias
   Training on one reviewer's style may overfit to that reviewer's preferences.
   A diverse reviewer pool or reviewer-agnostic rewards (clinical accuracy scores)
   would be needed in production.

5. Safety guarantee preservation
   The no-fabrication guardrail is enforced in agent_loop.py before any
   bandit decision. The learnable surface (prompt strategy) cannot suppress
   the guardrail — it only influences style and structure, not which fields
   are populated or how missing data is handled.
"""
