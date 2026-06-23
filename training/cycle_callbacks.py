from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CycleDecision:
    save_checkpoint: bool = False
    reduce_learning_rate: bool = False
    stop_training: bool = False
    message: str = ""


class ValidationCycleController:
    """Cycle-level checkpoint, LR reduction, and early-stopping policy."""

    def __init__(
        self,
        *,
        early_stopping_patience: int,
        early_stopping_min_cycles: int,
        early_stopping_min_delta: float,
        reduce_lr_patience: int,
    ) -> None:
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_cycles = early_stopping_min_cycles
        self.early_stopping_min_delta = early_stopping_min_delta
        self.reduce_lr_patience = reduce_lr_patience
        self.best_validation_f1 = float("-inf")
        self.plateau_reference_f1 = float("-inf")
        self.best_validation_loss = float("inf")
        self.f1_plateau_cycles = 0
        self.loss_plateau_cycles = 0
        self.awaiting_post_reduction_improvement = False
        self.post_reduction_plateau_cycles = 0

    def update(
        self,
        *,
        cycle: int,
        train_accuracy: float,
        validation_accuracy: float,
        validation_f1: float,
        validation_loss: float,
    ) -> CycleDecision:
        messages: list[str] = []
        decision = CycleDecision()

        if validation_f1 > self.best_validation_f1:
            self.best_validation_f1 = validation_f1
            decision.save_checkpoint = True
            messages.append(f"new_best_val_f1={validation_f1:.6f}")

        meaningful_f1_improvement = (
            validation_f1
            > self.plateau_reference_f1 + self.early_stopping_min_delta
        )
        if meaningful_f1_improvement:
            self.plateau_reference_f1 = validation_f1
            self.f1_plateau_cycles = 0
            if self.awaiting_post_reduction_improvement:
                self.awaiting_post_reduction_improvement = False
                self.post_reduction_plateau_cycles = 0
                messages.append("post_lr_reduction_recovered")
        else:
            self.f1_plateau_cycles += 1
            messages.append(
                f"val_f1_plateau={self.f1_plateau_cycles}/"
                f"{self.early_stopping_patience}"
            )
            if self.awaiting_post_reduction_improvement:
                self.post_reduction_plateau_cycles += 1
                messages.append(
                    "post_lr_plateau="
                    f"{self.post_reduction_plateau_cycles}/"
                    f"{self.early_stopping_patience}"
                )

        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.loss_plateau_cycles = 0
        else:
            self.loss_plateau_cycles += 1
            if self.loss_plateau_cycles >= self.reduce_lr_patience:
                decision.reduce_learning_rate = True
                self.loss_plateau_cycles = 0
                if not self.awaiting_post_reduction_improvement:
                    self.awaiting_post_reduction_improvement = True
                    self.post_reduction_plateau_cycles = 0
                messages.append("reduce_learning_rate")

        generalization_gap = train_accuracy - validation_accuracy
        messages.append(f"accuracy_gap={generalization_gap:.6f}")
        if generalization_gap > 0.20:
            messages.append("warning=large_generalization_gap")

        if (
            cycle >= self.early_stopping_min_cycles
            and self.awaiting_post_reduction_improvement
            and self.post_reduction_plateau_cycles
            >= self.early_stopping_patience
        ):
            decision.stop_training = True
            messages.append(
                "early_stop=validation_f1_not_improving_after_lr_reduction"
            )

        decision.message = " ".join(messages)
        return decision
