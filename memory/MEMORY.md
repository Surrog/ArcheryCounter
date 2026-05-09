# Memory Index

- [feedback_run_all_tests.md](feedback_run_all_tests.md) — Always run both targetDetection and groundTruth test suites before declaring a change correct
- [project_test_architecture.md](project_test_architecture.md) — Tests use generated DB table + globalSetup for fast runs (<2min); structural test runs inline on 1 image
- [feedback_arrow_detection_priority.md](feedback_arrow_detection_priority.md) — Arrow detection: precision over recall; missing arrows is acceptable, false positives are not
- [project_detection_improvements.md](project_detection_improvements.md) — groundTruth 18→12 failing; R2 (ring[7] ratio clamp) + A4 (shaft mask within ring[9]) implemented; failure details and next steps
- [project_nn_diagnostic.md](project_nn_diagnostic.md) — Dense FP root cause (focal loss bias imbalance); λ=5 sparsity fix; current limit is training data (~181 images); best model recall=0.646, precision=0.077
