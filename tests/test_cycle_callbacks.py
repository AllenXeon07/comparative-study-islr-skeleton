from __future__ import annotations

import unittest

from training.cycle_callbacks import ValidationCycleController


class ValidationCycleControllerTests(unittest.TestCase):
    def test_stops_only_after_post_reduction_plateau(self) -> None:
        controller = ValidationCycleController(
            early_stopping_patience=2,
            early_stopping_min_cycles=3,
            early_stopping_min_delta=0.01,
            reduce_lr_patience=2,
        )

        first = controller.update(
            cycle=1,
            train_accuracy=0.10,
            validation_accuracy=0.08,
            validation_f1=0.05,
            validation_loss=5.0,
        )
        second = controller.update(
            cycle=2,
            train_accuracy=0.12,
            validation_accuracy=0.08,
            validation_f1=0.055,
            validation_loss=5.1,
        )
        third = controller.update(
            cycle=3,
            train_accuracy=0.15,
            validation_accuracy=0.08,
            validation_f1=0.056,
            validation_loss=5.2,
        )
        fourth = controller.update(
            cycle=4,
            train_accuracy=0.15,
            validation_accuracy=0.08,
            validation_f1=0.057,
            validation_loss=5.3,
        )
        fifth = controller.update(
            cycle=5,
            train_accuracy=0.15,
            validation_accuracy=0.08,
            validation_f1=0.058,
            validation_loss=5.4,
        )

        self.assertTrue(first.save_checkpoint)
        self.assertTrue(second.save_checkpoint)
        self.assertTrue(third.reduce_learning_rate)
        self.assertFalse(third.stop_training)
        self.assertFalse(fourth.stop_training)
        self.assertTrue(fifth.stop_training)

    def test_resets_plateau_after_meaningful_f1_improvement(self) -> None:
        controller = ValidationCycleController(
            early_stopping_patience=2,
            early_stopping_min_cycles=3,
            early_stopping_min_delta=0.001,
            reduce_lr_patience=2,
        )
        controller.update(
            cycle=1,
            train_accuracy=0.10,
            validation_accuracy=0.08,
            validation_f1=0.05,
            validation_loss=5.0,
        )
        result = controller.update(
            cycle=2,
            train_accuracy=0.12,
            validation_accuracy=0.10,
            validation_f1=0.06,
            validation_loss=4.9,
        )

        self.assertTrue(result.save_checkpoint)
        self.assertFalse(result.stop_training)

    def test_saves_small_improvement_without_resetting_plateau(self) -> None:
        controller = ValidationCycleController(
            early_stopping_patience=3,
            early_stopping_min_cycles=4,
            early_stopping_min_delta=0.01,
            reduce_lr_patience=2,
        )
        controller.update(
            cycle=1,
            train_accuracy=0.10,
            validation_accuracy=0.08,
            validation_f1=0.05,
            validation_loss=5.0,
        )
        result = controller.update(
            cycle=2,
            train_accuracy=0.11,
            validation_accuracy=0.09,
            validation_f1=0.055,
            validation_loss=4.9,
        )

        self.assertTrue(result.save_checkpoint)
        self.assertEqual(controller.f1_plateau_cycles, 1)

    def test_meaningful_improvement_after_reduction_cancels_stop_window(self) -> None:
        controller = ValidationCycleController(
            early_stopping_patience=2,
            early_stopping_min_cycles=3,
            early_stopping_min_delta=0.01,
            reduce_lr_patience=2,
        )
        controller.update(
            cycle=1,
            train_accuracy=0.10,
            validation_accuracy=0.08,
            validation_f1=0.05,
            validation_loss=5.0,
        )
        controller.update(
            cycle=2,
            train_accuracy=0.12,
            validation_accuracy=0.08,
            validation_f1=0.055,
            validation_loss=5.1,
        )
        reduction = controller.update(
            cycle=3,
            train_accuracy=0.14,
            validation_accuracy=0.09,
            validation_f1=0.056,
            validation_loss=5.2,
        )
        recovery = controller.update(
            cycle=4,
            train_accuracy=0.15,
            validation_accuracy=0.11,
            validation_f1=0.07,
            validation_loss=5.0,
        )

        self.assertTrue(reduction.reduce_learning_rate)
        self.assertFalse(recovery.stop_training)
        self.assertFalse(controller.awaiting_post_reduction_improvement)


if __name__ == "__main__":
    unittest.main()
