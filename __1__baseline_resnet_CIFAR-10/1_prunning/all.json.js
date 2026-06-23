// ────────────────────────────────────────────────
// DATA — embedded from all 7 JSON files
// ────────────────────────────────────────────────
const DATA = {
    unstructured: {
        method: "unstructured_pruning",
        description: "Global L1 unstructured pruning \u2014 individual weights zeroed",
        pruning_granularity: "weight",
        baseline: { accuracy: 0.932, precision: 0.9318626541334971, recall: 0.9319999999999998, f1: 0.9318835910591599, params: 23520842, size_mb: 94.408905, flops_G: 1.311594496, inference_ms: 4.578483581542969 },
        baseline_size_breakdown: { dense_ram_mb: 89.7249, disk_pth_mb: 90.0294 },
        pruning_criterion: "L1 magnitude (|w|)",
        device: "cuda",
        max_acc_drop_threshold: 0.02,
        best_sparsity: 0.3,
        results: [
            { target_sparsity: 0.3, actual_sparsity: 0.3, accuracy: 0.9321, precision: 0.931969, recall: 0.9321, f1: 0.931987, accuracy_drop: -0.0001, params_total: 23520842, params_active: 16480528, size_dense_mb: 89.7249, size_sparse_theoretical_mb: 62.8682, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 142.0333, inference_cpu_ms: 3531.2448 },
            { target_sparsity: 0.5, actual_sparsity: 0.5, accuracy: 0.932, precision: 0.93186, recall: 0.932, f1: 0.931878, accuracy_drop: 0.0, params_total: 23520842, params_active: 11786986, size_dense_mb: 89.7249, size_sparse_theoretical_mb: 44.9638, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 142.205, inference_cpu_ms: 3554.5796 },
            { target_sparsity: 0.7, actual_sparsity: 0.7, accuracy: 0.9319, precision: 0.931749, recall: 0.9319, f1: 0.931752, accuracy_drop: 0.0001, params_total: 23520842, params_active: 7093444, size_dense_mb: 89.7249, size_sparse_theoretical_mb: 27.0593, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 140.535, inference_cpu_ms: 3971.5077 },
            { target_sparsity: 0.8, actual_sparsity: 0.8, accuracy: 0.9276, precision: 0.927693, recall: 0.9276, f1: 0.927507, accuracy_drop: 0.0044, params_total: 23520842, params_active: 4746672, size_dense_mb: 89.7249, size_sparse_theoretical_mb: 18.1071, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.9209, inference_cpu_ms: 3122.884 },
            { target_sparsity: 0.9, actual_sparsity: 0.9, accuracy: 0.8774, precision: 0.88906, recall: 0.8774, f1: 0.878868, accuracy_drop: 0.0546, params_total: 23520842, params_active: 2399901, size_dense_mb: 89.7249, size_sparse_theoretical_mb: 9.1549, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.8964, inference_cpu_ms: 3111.5664 }
        ],
        notes: {
            size_dense_mb: "Full float32 RAM \u2014 never shrinks",
            size_sparse_theoretical_mb: "active_params * 4B \u2014 sparse format lower bound",
            size_disk_mb: "Actual .pth (compressed zip)",
            gpu_speedup: "Dense CUDA does NOT skip zeros \u2014 no GPU speedup without cuSPARSE"
        }
    },

    structured: {
        method: "structured_pruning",
        description: "L1 structured filter pruning \u2014 entire Conv2d output filters removed",
        pruning_granularity: "filter (output channel, dim=0)",
        baseline: { accuracy: 0.932, precision: 0.9318626541334971, recall: 0.9319999999999998, f1: 0.9318835910591599, params: 23520842, size_mb: 94.408905, flops_G: 1.311594496, inference_ms: 4.578483581542969 },
        baseline_size_breakdown: { dense_ram_mb: 89.7249, disk_pth_mb: 90.0294 },
        pruning_criterion: "L1 norm of filter (sum of |w| per output filter)",
        device: "cuda",
        max_acc_drop_threshold: 0.02,
        best_ratio: 0.1,
        results: [
            { filter_pruning_ratio: 0.1, structured_sparsity: 0.099, weight_sparsity: 0.0989, accuracy: 0.9322, precision: 0.932111, recall: 0.9322, f1: 0.932104, accuracy_drop: -0.0002, params_total: 23520842, params_active: 21200808, size_dense_mb: 89.7249, size_sparse_theoretical_mb: 80.8747, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 221.2131, inference_cpu_ms: 3475.5928 },
            { filter_pruning_ratio: 0.2, structured_sparsity: 0.199, weight_sparsity: 0.1989, accuracy: 0.931, precision: 0.931048, recall: 0.931, f1: 0.930961, accuracy_drop: 0.001, params_total: 23520842, params_active: 18852102, size_dense_mb: 89.7249, size_sparse_theoretical_mb: 71.9151, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 51.4466, inference_cpu_ms: 3414.9131 },
            { filter_pruning_ratio: 0.3, structured_sparsity: 0.299, weight_sparsity: 0.2984, accuracy: 0.9313, precision: 0.931521, recall: 0.9313, f1: 0.931253, accuracy_drop: 0.0007, params_total: 23520842, params_active: 16518729, size_dense_mb: 89.7249, size_sparse_theoretical_mb: 63.014, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 299.1997, inference_cpu_ms: 3251.3643 },
            { filter_pruning_ratio: 0.4, structured_sparsity: 0.399, weight_sparsity: 0.3985, accuracy: 0.9227, precision: 0.923856, recall: 0.9227, f1: 0.922834, accuracy_drop: 0.0093, params_total: 23520842, params_active: 14170023, size_dense_mb: 89.7249, size_sparse_theoretical_mb: 54.0543, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 51.0766, inference_cpu_ms: 3353.3515 },
            { filter_pruning_ratio: 0.5, structured_sparsity: 0.5, weight_sparsity: 0.4996, accuracy: 0.8734, precision: 0.887076, recall: 0.8734, f1: 0.874671, accuracy_drop: 0.0586, params_total: 23520842, params_active: 11797226, size_dense_mb: 89.7249, size_sparse_theoretical_mb: 45.0028, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 195.7928, inference_cpu_ms: 3324.7461 }
        ],
        notes: {
            structured_sparsity: "Fraction of zero-norm filters in Conv2d layers",
            weight_sparsity: "Fraction of zero weight values (all weights in zero filters)",
            real_speedup: "Structured pruning CAN yield real speedup after model rebuild \u2014 zero-row Conv ops are skipped by cuDNN. PyTorch's prune API zeroes rows but keeps tensor shapes; rebuild for actual FLOP reduction.",
            size_note: "Disk size compresses well due to zero-filter rows"
        }
    },

    magnitude: {
        method: "magnitude_pruning",
        description: "Per-layer magnitude pruning comparing L1-local, L2-local, and global-L1",
        pruning_granularity: "weight (unstructured)",
        baseline: { accuracy: 0.932, precision: 0.9318626541334971, recall: 0.9319999999999998, f1: 0.9318835910591599, params: 23520842, size_mb: 94.408905, flops_G: 1.311594496, inference_ms: 4.578483581542969 },
        pruning_criteria: [
            "local_l1 (per-layer L1)",
            "local_l2 (per-layer L2^2)",
            "global_l1 (all layers ranked together)"
        ],
        device: "cuda",
        max_acc_drop_threshold: 0.02,
        best_config: {
            sparsity: 0.3,
            criterion: "global_l1"
        },
        results: [
            { target_sparsity: 0.3, criterion: "local_l1", actual_sparsity: 0.3, accuracy: 0.9319, precision: 0.931712, recall: 0.9319, f1: 0.931759, accuracy_drop: 0.0001, params_total: 23520842, params_active: 16480530, size_sparse_theoretical_mb: 62.8682, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 57.9127, inference_cpu_ms: 5425.1179 },
            { target_sparsity: 0.3, criterion: "local_l2", actual_sparsity: 0.3, accuracy: 0.9319, precision: 0.931712, recall: 0.9319, f1: 0.931759, accuracy_drop: 0.0001, params_total: 23520842, params_active: 16480553, size_sparse_theoretical_mb: 62.8683, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 53.2483, inference_cpu_ms: 3290.6214 },
            { target_sparsity: 0.3, criterion: "global_l1", actual_sparsity: 0.3, accuracy: 0.9321, precision: 0.931969, recall: 0.9321, f1: 0.931987, accuracy_drop: -0.0001, params_total: 23520842, params_active: 16480528, size_sparse_theoretical_mb: 62.8682, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 51.0954, inference_cpu_ms: 3277.1548 },
            { target_sparsity: 0.5, criterion: "local_l1", actual_sparsity: 0.5, accuracy: 0.9318, precision: 0.931473, recall: 0.9318, f1: 0.931555, accuracy_drop: 0.0002, params_total: 23520842, params_active: 11786986, size_sparse_theoretical_mb: 44.9638, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 51.014, inference_cpu_ms: 3267.0098 },
            { target_sparsity: 0.5, criterion: "local_l2", actual_sparsity: 0.5, accuracy: 0.9318, precision: 0.931473, recall: 0.9318, f1: 0.931555, accuracy_drop: 0.0002, params_total: 23520842, params_active: 11786986, size_sparse_theoretical_mb: 44.9638, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 76.4604, inference_cpu_ms: 3233.5946 },
            { target_sparsity: 0.5, criterion: "global_l1", actual_sparsity: 0.5, accuracy: 0.932, precision: 0.93186, recall: 0.932, f1: 0.931878, accuracy_drop: 0.0, params_total: 23520842, params_active: 11786986, size_sparse_theoretical_mb: 44.9638, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 51.0497, inference_cpu_ms: 3115.1571 },
            { target_sparsity: 0.7, criterion: "local_l1", actual_sparsity: 0.7, accuracy: 0.9286, precision: 0.928269, recall: 0.9286, f1: 0.928109, accuracy_drop: 0.0034, params_total: 23520842, params_active: 7093442, size_sparse_theoretical_mb: 27.0593, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.962, inference_cpu_ms: 3128.8472 },
            { target_sparsity: 0.7, criterion: "local_l2", actual_sparsity: 0.7, accuracy: 0.9286, precision: 0.928269, recall: 0.9286, f1: 0.928109, accuracy_drop: 0.0034, params_total: 23520842, params_active: 7093472, size_sparse_theoretical_mb: 27.0594, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.9619, inference_cpu_ms: 3142.9595 },
            { target_sparsity: 0.7, criterion: "global_l1", actual_sparsity: 0.7, accuracy: 0.9319, precision: 0.931749, recall: 0.9319, f1: 0.931752, accuracy_drop: 0.0001, params_total: 23520842, params_active: 7093444, size_sparse_theoretical_mb: 27.0593, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 106.0588, inference_cpu_ms: 3078.3612 },
            { target_sparsity: 0.8, criterion: "local_l1", actual_sparsity: 0.8, accuracy: 0.9155, precision: 0.91625, recall: 0.9155, f1: 0.914683, accuracy_drop: 0.0165, params_total: 23520842, params_active: 4746674, size_sparse_theoretical_mb: 18.1071, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.9118, inference_cpu_ms: 3307.7786 },
            { target_sparsity: 0.8, criterion: "local_l2", actual_sparsity: 0.8, accuracy: 0.9154, precision: 0.916157, recall: 0.9154, f1: 0.914574, accuracy_drop: 0.0166, params_total: 23520842, params_active: 4746697, size_sparse_theoretical_mb: 18.1072, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.9127, inference_cpu_ms: 3069.3876 },
            { target_sparsity: 0.8, criterion: "global_l1", actual_sparsity: 0.8, accuracy: 0.9276, precision: 0.927693, recall: 0.9276, f1: 0.927507, accuracy_drop: 0.0044, params_total: 23520842, params_active: 4746672, size_sparse_theoretical_mb: 18.1071, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.9577, inference_cpu_ms: 3108.3937 },
            { target_sparsity: 0.9, criterion: "local_l1", actual_sparsity: 0.9, accuracy: 0.689, precision: 0.782957, recall: 0.689, f1: 0.690602, accuracy_drop: 0.243, params_total: 23520842, params_active: 2399899, size_sparse_theoretical_mb: 9.1549, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.8749, inference_cpu_ms: 3466.0524 },
            { target_sparsity: 0.9, criterion: "local_l2", actual_sparsity: 0.9, accuracy: 0.6888, precision: 0.782728, recall: 0.6888, f1: 0.690348, accuracy_drop: 0.2432, params_total: 23520842, params_active: 2399928, size_sparse_theoretical_mb: 9.155, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.9332, inference_cpu_ms: 3260.5361 },
            { target_sparsity: 0.9, criterion: "global_l1", actual_sparsity: 0.9, accuracy: 0.8774, precision: 0.88906, recall: 0.8774, f1: 0.878868, accuracy_drop: 0.0546, params_total: 23520842, params_active: 2399901, size_sparse_theoretical_mb: 9.1549, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.8726, inference_cpu_ms: 3112.5039 }
        ],
        notes: {
            local_l1: "Each layer independently pruned to target sparsity \u2014 uniform compression",
            local_l2: "Each layer pruned by L2^2 score \u2014 prefers keeping large-magnitude weights",
            global_l1: "All layers ranked together \u2014 important layers auto-protected",
            comparison_insight: "Global L1 typically outperforms local methods at high sparsity"
        }
    },

    movement: {
        method: "movement_pruning",
        description: "Movement pruning via Taylor importance |w*grad| \u2014 data-aware pruning",
        pruning_granularity: "weight (unstructured)",
        scoring_method: "Taylor first-order: |weight * gradient|",
        calibration_batches: 10,
        baseline: {
            accuracy: 0.932, precision: 0.9318626541334971, recall: 0.9319999999999998, f1: 0.9318835910591599,
            params: 23520842,
            size_mb: 94.408905,
            flops_G: 1.311594496,
            inference_ms: 4.578483581542969
        },
        device: "cuda",
        max_acc_drop_threshold: 0.02,
        best_sparsity: 0.3,
        results: [
            { target_sparsity: 0.3, actual_sparsity: 0.3209, accuracy: 0.928, precision: 0.92794, recall: 0.928, f1: 0.927868, accuracy_drop: 0.004, params_total: 23520842, params_active: 15989963, size_sparse_theoretical_mb: 60.9969, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 51.5753, inference_cpu_ms: 3418.9204, calibration_batches_used: 10 },
            { target_sparsity: 0.5, actual_sparsity: 0.5, accuracy: 0.9271, precision: 0.927032, recall: 0.9271, f1: 0.926943, accuracy_drop: 0.0049, params_total: 23520842, params_active: 11786986, size_sparse_theoretical_mb: 44.9638, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 1118.2021, inference_cpu_ms: 3362.2209, calibration_batches_used: 10 },
            { target_sparsity: 0.7, actual_sparsity: 0.7, accuracy: 0.9222, precision: 0.922995, recall: 0.9222, f1: 0.922136, accuracy_drop: 0.0098, params_total: 23520842, params_active: 7093444, size_sparse_theoretical_mb: 27.0593, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.9643, inference_cpu_ms: 3289.8048, calibration_batches_used: 10 },
            { target_sparsity: 0.8, actual_sparsity: 0.8, accuracy: 0.895, precision: 0.900283, recall: 0.895, f1: 0.89551, accuracy_drop: 0.037, params_total: 23520842, params_active: 4746673, size_sparse_theoretical_mb: 18.1071, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.8915, inference_cpu_ms: 3160.901, calibration_batches_used: 10 },
            { target_sparsity: 0.9, actual_sparsity: 0.9, accuracy: 0.5335, precision: 0.767764, recall: 0.5335, f1: 0.518966, accuracy_drop: 0.3985, params_total: 23520842, params_active: 2399902, size_sparse_theoretical_mb: 9.1549, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 194.095, inference_cpu_ms: 3101.5054, calibration_batches_used: 10 }
        ],
        notes: {
            theory: "True movement pruning trains soft masks alongside weights. Here we use Taylor importance (|w*grad|) as a one-shot post-training approximation.",
            advantage_over_magnitude: "Data-aware \u2014 considers gradient signal, not just weight value. Small weights with large gradients are preserved.",
            limitation: "One-shot Taylor approximation; full movement pruning requires training with soft masks (see Sanh et al. 2020)"
        }
    },

    lottery: {
        method: "lottery_ticket_hypothesis",
        description: "LTH: iterative magnitude pruning to find ticket mask; compare trained-weights vs random-init",
        pruning_granularity: "weight (unstructured)",
        iterative_rounds: 5,
        baseline: {
            accuracy: 0.932, precision: 0.9318626541334971, recall: 0.9319999999999998, f1: 0.9318835910591599,
            params: 23520842,
            size_mb: 94.408905,
            flops_G: 1.311594496,
            inference_ms: 4.578483581542969
        },
        device: "cuda",
        max_acc_drop_threshold: 0.02,
        best_sparsity: 0.3,
        results: [
            { target_sparsity: 0.3, actual_sparsity: 0.0689, iterative_rounds: 5, winning_ticket_accuracy: 0.9321, winning_ticket_precision: 0.931969, winning_ticket_recall: 0.9321, winning_ticket_f1: 0.931987, winning_ticket_accuracy_drop: -0.0001, random_init_accuracy: 0.1, random_init_accuracy_drop: 0.832, lth_advantage_pct: 83.21, params_total: 23520842, params_active: 21905088, size_sparse_theoretical_mb: 83.5613, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 193.9307, inference_cpu_ms: 3519.8752 },
            { target_sparsity: 0.5, actual_sparsity: 0.1294, iterative_rounds: 5, winning_ticket_accuracy: 0.9321, winning_ticket_precision: 0.931969, winning_ticket_recall: 0.9321, winning_ticket_f1: 0.931987, winning_ticket_accuracy_drop: -0.0001, random_init_accuracy: 0.1, random_init_accuracy_drop: 0.832, lth_advantage_pct: 83.21, params_total: 23520842, params_active: 20482960, size_sparse_theoretical_mb: 78.1363, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 52.0958, inference_cpu_ms: 3371.8594 },
            { target_sparsity: 0.7, actual_sparsity: 0.214, iterative_rounds: 5, winning_ticket_accuracy: 0.9321, winning_ticket_precision: 0.931969, winning_ticket_recall: 0.9321, winning_ticket_f1: 0.931987, winning_ticket_accuracy_drop: -0.0001, random_init_accuracy: 0.1, random_init_accuracy_drop: 0.832, lth_advantage_pct: 83.21, params_total: 23520842, params_active: 18498824, size_sparse_theoretical_mb: 70.5674, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 249.047, inference_cpu_ms: 3367.0707 },
            { target_sparsity: 0.8, actual_sparsity: 0.2752, iterative_rounds: 5, winning_ticket_accuracy: 0.9321, winning_ticket_precision: 0.931969, winning_ticket_recall: 0.9321, winning_ticket_f1: 0.931987, winning_ticket_accuracy_drop: -0.0001, random_init_accuracy: 0.1175, random_init_accuracy_drop: 0.8145, lth_advantage_pct: 81.46, params_total: 23520842, params_active: 17062050, size_sparse_theoretical_mb: 65.0866, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 256.5479, inference_cpu_ms: 3325.454 },
            { target_sparsity: 0.9, actual_sparsity: 0.369, iterative_rounds: 5, winning_ticket_accuracy: 0.9321, winning_ticket_precision: 0.931969, winning_ticket_recall: 0.9321, winning_ticket_f1: 0.931987, winning_ticket_accuracy_drop: -0.0001, random_init_accuracy: 0.1, random_init_accuracy_drop: 0.832, lth_advantage_pct: 83.21, params_total: 23520842, params_active: 14860255, size_sparse_theoretical_mb: 56.6874, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 61.7101, inference_cpu_ms: 3286.0964 }
        ],
        notes: {
            winning_ticket: "Ticket mask + trained (\u03b8_T) weights. Represents pruned baseline.",
            random_init: "Same mask + kaiming-normal re-initialized weights. Control: untrained subnetwork.",
            lth_advantage_pct: "Positive = trained weights preserve accuracy vs random init. Validates LTH.",
            full_lth: "True LTH trains the re-initialized subnetwork; here we only evaluate it untrained as a demonstration.",
            reference: "Frankle & Carlin (2019) https://arxiv.org/abs/1803.03635"
        }
    },

    iterative: {
        method: "iterative_pruning",
        description: "Iterative L1 magnitude pruning: 15% of remaining weights per round",
        pruning_granularity: "weight (unstructured, global L1)",
        pruning_step: 0.15,
        total_rounds: 15,
        baseline: {
            accuracy: 0.932, precision: 0.9318626541334971, recall: 0.9319999999999998, f1: 0.9318835910591599,
            params: 23520842,
            size_mb: 94.408905,
            flops_G: 1.311594496,
            inference_ms: 4.578483581542969
        },
        device: "cuda",
        max_acc_drop_threshold: 0.02,
        best_sparsity: 0.6229,
        trajectory: [
            { round: 1, pruning_step_fraction: 0.15, cumulative_sparsity: 0.15, accuracy: 0.9321, precision: 0.931969, recall: 0.9321, f1: 0.931987, accuracy_drop: -0.0001, params_total: 23520842, params_active: 20000686, size_sparse_theoretical_mb: 76.2966, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 51.3008, inference_cpu_ms: 6143.746 },
            { round: 2, pruning_step_fraction: 0.15, cumulative_sparsity: 0.2775, accuracy: 0.9321, precision: 0.931969, recall: 0.9321, f1: 0.931987, accuracy_drop: -0.0001, params_total: 23520842, params_active: 17008553, size_sparse_theoretical_mb: 64.8825, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 51.0015, inference_cpu_ms: 3249.7135 },
            { round: 3, pruning_step_fraction: 0.15, cumulative_sparsity: 0.3859, accuracy: 0.932, precision: 0.93187, recall: 0.932, f1: 0.931886, accuracy_drop: 0.0, params_total: 23520842, params_active: 14465239, size_sparse_theoretical_mb: 55.1805, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 51.1445, inference_cpu_ms: 3328.8828 },
            { round: 4, pruning_step_fraction: 0.15, cumulative_sparsity: 0.478, accuracy: 0.9321, precision: 0.931962, recall: 0.9321, f1: 0.93198, accuracy_drop: -0.0001, params_total: 23520842, params_active: 12303423, size_sparse_theoretical_mb: 46.9338, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 51.1446, inference_cpu_ms: 3279.7643 },
            { round: 5, pruning_step_fraction: 0.15, cumulative_sparsity: 0.5563, accuracy: 0.932, precision: 0.931863, recall: 0.932, f1: 0.931883, accuracy_drop: 0.0, params_total: 23520842, params_active: 10465879, size_sparse_theoretical_mb: 39.9242, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 51.1174, inference_cpu_ms: 3311.6608 },
            { round: 6, pruning_step_fraction: 0.15, cumulative_sparsity: 0.6229, accuracy: 0.9322, precision: 0.932076, recall: 0.9322, f1: 0.932078, accuracy_drop: -0.0002, params_total: 23520842, params_active: 8903967, size_sparse_theoretical_mb: 33.9659, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 112.216, inference_cpu_ms: 3296.7424 },
            { round: 7, pruning_step_fraction: 0.15, cumulative_sparsity: 0.6794, accuracy: 0.932, precision: 0.931827, recall: 0.932, f1: 0.931851, accuracy_drop: 0.0, params_total: 23520842, params_active: 7576341, size_sparse_theoretical_mb: 28.9014, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 51.0572, inference_cpu_ms: 3270.1927 },
            { round: 8, pruning_step_fraction: 0.15, cumulative_sparsity: 0.7275, accuracy: 0.9313, precision: 0.931167, recall: 0.9313, f1: 0.931159, accuracy_drop: 0.0007, params_total: 23520842, params_active: 6447859, size_sparse_theoretical_mb: 24.5966, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 51.1588, inference_cpu_ms: 3278.0833 },
            { round: 9, pruning_step_fraction: 0.15, cumulative_sparsity: 0.7684, accuracy: 0.9302, precision: 0.930126, recall: 0.9302, f1: 0.930067, accuracy_drop: 0.0018, params_total: 23520842, params_active: 5488650, size_sparse_theoretical_mb: 20.9375, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.9542, inference_cpu_ms: 3129.8862 },
            { round: 10, pruning_step_fraction: 0.15, cumulative_sparsity: 0.8031, accuracy: 0.9283, precision: 0.928397, recall: 0.9283, f1: 0.928199, accuracy_drop: 0.0037, params_total: 23520842, params_active: 4673322, size_sparse_theoretical_mb: 17.8273, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.9289, inference_cpu_ms: 3124.721 },
            { round: 11, pruning_step_fraction: 0.15, cumulative_sparsity: 0.8327, accuracy: 0.924, precision: 0.924727, recall: 0.924, f1: 0.924042, accuracy_drop: 0.008, params_total: 23520842, params_active: 3980293, size_sparse_theoretical_mb: 15.1836, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.8718, inference_cpu_ms: 3125.5835 },
            { round: 12, pruning_step_fraction: 0.15, cumulative_sparsity: 0.8578, accuracy: 0.9193, precision: 0.920636, recall: 0.9193, f1: 0.919437, accuracy_drop: 0.0127, params_total: 23520842, params_active: 3391219, size_sparse_theoretical_mb: 12.9365, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.9088, inference_cpu_ms: 3103.9854 },
            { round: 13, pruning_step_fraction: 0.15, cumulative_sparsity: 0.8791, accuracy: 0.9058, precision: 0.909335, recall: 0.9058, f1: 0.906272, accuracy_drop: 0.0262, params_total: 23520842, params_active: 2890506, size_sparse_theoretical_mb: 11.0264, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.8462, inference_cpu_ms: 3101.739 },
            { round: 14, pruning_step_fraction: 0.15, cumulative_sparsity: 0.8972, accuracy: 0.8838, precision: 0.893621, recall: 0.8838, f1: 0.885013, accuracy_drop: 0.0482, params_total: 23520842, params_active: 2464900, size_sparse_theoretical_mb: 9.4028, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.9205, inference_cpu_ms: 3144.241 },
            { round: 15, pruning_step_fraction: 0.15, cumulative_sparsity: 0.9126, accuracy: 0.8496, precision: 0.873072, recall: 0.8496, f1: 0.852351, accuracy_drop: 0.0824, params_total: 23520842, params_active: 2103135, size_sparse_theoretical_mb: 8.0228, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 102.0214, inference_cpu_ms: 3122.0633 }
        ],
        checkpointed_results: [
            { round: 2, pruning_step_fraction: 0.15, cumulative_sparsity: 0.2775, accuracy: 0.9321, precision: 0.931969, recall: 0.9321, f1: 0.931987, accuracy_drop: -0.0001, params_total: 23520842, params_active: 17008553, size_sparse_theoretical_mb: 64.8825, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 51.0015, inference_cpu_ms: 3249.7135, Fcheckpoint_target: 0.3 },
            { round: 4, pruning_step_fraction: 0.15, cumulative_sparsity: 0.478, accuracy: 0.9321, precision: 0.931962, recall: 0.9321, f1: 0.93198, accuracy_drop: -0.0001, params_total: 23520842, params_active: 12303423, size_sparse_theoretical_mb: 46.9338, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 51.1446, inference_cpu_ms: 3279.7643, Fcheckpoint_target: 0.5 },
            { round: 7, pruning_step_fraction: 0.15, cumulative_sparsity: 0.6794, accuracy: 0.932, precision: 0.931827, recall: 0.932, f1: 0.931851, accuracy_drop: 0.0, params_total: 23520842, params_active: 7576341, size_sparse_theoretical_mb: 28.9014, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 51.0572, inference_cpu_ms: 3270.1927, Fcheckpoint_target: 0.7 },
            { round: 10, pruning_step_fraction: 0.15, cumulative_sparsity: 0.8031, accuracy: 0.9283, precision: 0.928397, recall: 0.9283, f1: 0.928199, accuracy_drop: 0.0037, params_total: 23520842, params_active: 4673322, size_sparse_theoretical_mb: 17.8273, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.9289, inference_cpu_ms: 3124.721, Fcheckpoint_target: 0.8 },
            { round: 14, pruning_step_fraction: 0.15, cumulative_sparsity: 0.8972, accuracy: 0.8838, precision: 0.893621, recall: 0.8838, f1: 0.885013, accuracy_drop: 0.0482, params_total: 23520842, params_active: 2464900, size_sparse_theoretical_mb: 9.4028, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.9205, inference_cpu_ms: 3144.241, Fcheckpoint_target: 0.9 }
        ],
        notes: {
            trajectory: "Full pruning trajectory: accuracy at each incremental round",
            advantage: "Reveals where accuracy begins to drop, not just final value",
            vs_one_shot: "Compare with 7_oneshot_Pruning.json to see iterative vs one-shot difference",
            step_formula: "Each round removes 15% of REMAINING non-zero weights (not of total)"
        }
    },

    oneshot: {
        method: "oneshot_pruning",
        description: "One-shot pruning: target sparsity reached in a single step \u2014 no iteration",
        pruning_granularity: "weight (unstructured, global)",
        variants: [
            "oneshot_l1_global",
            "oneshot_l2_global",
            "oneshot_random"
        ],
        baseline: { accuracy: 0.932, precision: 0.9318626541334971, recall: 0.9319999999999998, f1: 0.9318835910591599, params: 23520842, size_mb: 94.408905, flops_G: 1.311594496, inference_ms: 4.578483581542969 },
        device: "cuda",
        max_acc_drop_threshold: 0.02,
        best_config: { sparsity: 0.3, variant: "oneshot_l1_global" },
        results: [
            { target_sparsity: 0.3, variant: "oneshot_l1_global", actual_sparsity: 0.3, accuracy: 0.9321, precision: 0.931969, recall: 0.9321, f1: 0.931987, accuracy_drop: -0.0001, params_total: 23520842, params_active: 16480528, size_sparse_theoretical_mb: 62.8682, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 194.6976, inference_cpu_ms: 3495.346 },
            { target_sparsity: 0.3, variant: "oneshot_random", actual_sparsity: 0.3, accuracy: 0.1001, precision: 0.110001, recall: 0.1001, f1: 0.018383, accuracy_drop: 0.8319, params_total: 23520842, params_active: 16480528, size_sparse_theoretical_mb: 62.8682, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 55.5177, inference_cpu_ms: 3350.3458 },
            { target_sparsity: 0.5, variant: "oneshot_l1_global", actual_sparsity: 0.5, accuracy: 0.932, precision: 0.93186, recall: 0.932, f1: 0.931878, accuracy_drop: 0.0, params_total: 23520842, params_active: 11786986, size_sparse_theoretical_mb: 44.9638, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 206.405, inference_cpu_ms: 3290.7732 },
            { target_sparsity: 0.5, variant: "oneshot_random", actual_sparsity: 0.5, accuracy: 0.1, precision: 0.01, recall: 0.1, f1: 0.018182, accuracy_drop: 0.832, params_total: 23520842, params_active: 11786986, size_sparse_theoretical_mb: 44.9638, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 51.003, inference_cpu_ms: 3263.4111 },
            { target_sparsity: 0.7, variant: "oneshot_l1_global", actual_sparsity: 0.7, accuracy: 0.9319, precision: 0.931749, recall: 0.9319, f1: 0.931752, accuracy_drop: 0.0001, params_total: 23520842, params_active: 7093444, size_sparse_theoretical_mb: 27.0593, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 163.2746, inference_cpu_ms: 3313.0667 },
            { target_sparsity: 0.7, variant: "oneshot_random", actual_sparsity: 0.7, accuracy: 0.1, precision: 0.01, recall: 0.1, f1: 0.018182, accuracy_drop: 0.832, params_total: 23520842, params_active: 7093444, size_sparse_theoretical_mb: 27.0593, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.9671, inference_cpu_ms: 3269.5682 },
            { target_sparsity: 0.8, variant: "oneshot_l1_global", actual_sparsity: 0.8, accuracy: 0.9276, precision: 0.927693, recall: 0.9276, f1: 0.927507, accuracy_drop: 0.0044, params_total: 23520842, params_active: 4746672, size_sparse_theoretical_mb: 18.1071, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.929, inference_cpu_ms: 3109.8385 },
            { target_sparsity: 0.8, variant: "oneshot_random", actual_sparsity: 0.8, accuracy: 0.1, precision: 0.01, recall: 0.1, f1: 0.018182, accuracy_drop: 0.832, params_total: 23520842, params_active: 4746672, size_sparse_theoretical_mb: 18.1071, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.908, inference_cpu_ms: 3104.3672 },
            { target_sparsity: 0.9, variant: "oneshot_l1_global", actual_sparsity: 0.9, accuracy: 0.8774, precision: 0.88906, recall: 0.8774, f1: 0.878868, accuracy_drop: 0.0546, params_total: 23520842, params_active: 2399901, size_sparse_theoretical_mb: 9.1549, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.8833, inference_cpu_ms: 3215.56 },
            { target_sparsity: 0.9, variant: "oneshot_random", actual_sparsity: 0.9, accuracy: 0.1, precision: 0.01, recall: 0.1, f1: 0.018182, accuracy_drop: 0.832, params_total: 23520842, params_active: 2399901, size_sparse_theoretical_mb: 9.1549, size_disk_mb: 90.0294, disk_saved_mb: 0.0, inference_gpu_ms: 50.8289, inference_cpu_ms: 3127.0789 }
        ],
        notes: {
            oneshot_l1_global: "Global L1 ranking, single pass \u2014 same as unstructured script",
            oneshot_l2_global: "Global L2 (squared magnitude) ranking, single pass",
            oneshot_random: "Randomly selected weights pruned \u2014 control baseline, worst accuracy",
            vs_iterative: "Compare with 6_iterative_Pruning.json to see cost of one-shot vs iterative",
            speed: "One-shot is extremely fast \u2014 no loops, single threshold computation"
        }
    }
};