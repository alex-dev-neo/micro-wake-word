# coding=utf-8
# Copyright 2023 The Google Research Authors.
# Modifications copyright 2024 Kevin Ahrendt.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import contextlib
import multiprocessing
import queue
import time

from absl import logging

import numpy as np
import tensorflow as tf
# Import worker function from data.py
import microwakeword.data as data_module 

from tensorflow.python.util import tf_decorator


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


@contextlib.contextmanager
def swap_attribute(obj, attr, temp_value):
    """Temporarily swap an attribute of an object."""
    original_value = getattr(obj, attr)
    setattr(obj, attr, temp_value)
    try:
        yield
    finally:
        setattr(obj, attr, original_value)


def validate_nonstreaming(config, data_processor, model, test_set):
    # Get all validation data
    testing_fingerprints, testing_ground_truth, _ = data_processor.get_data(
        test_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="truncate_start",
    )
    
    testing_ground_truth = testing_ground_truth.reshape(-1, 1)

    model.reset_metrics()

    # FIX OOM: Streaming validation
    test_ds = tf.data.Dataset.from_tensor_slices(
        (testing_fingerprints, testing_ground_truth)
    ).batch(128).prefetch(tf.data.AUTOTUNE)

    result = model.evaluate(test_ds, return_dict=True, verbose=0)

    metrics = {}
    metrics["accuracy"] = result["accuracy"]
    metrics["recall"] = result["recall"]
    metrics["precision"] = result["precision"]
    metrics["auc"] = result["auc"]
    metrics["loss"] = result["loss"]
    metrics["recall_at_no_faph"] = 0
    metrics["cutoff_for_no_faph"] = 0
    metrics["ambient_false_positives"] = 0
    metrics["ambient_false_positives_per_hour"] = 0
    metrics["average_viable_recall"] = 0

    test_set_fp = _to_numpy(result["fp"])

    if data_processor.get_mode_size("validation_ambient") > 0:
        (
            ambient_testing_fingerprints,
            ambient_testing_ground_truth,
            _,
        ) = data_processor.get_data(
            test_set + "_ambient",
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            truncation_strategy="split",
        )
        ambient_testing_ground_truth = ambient_testing_ground_truth.reshape(-1, 1)

        ambient_ds = tf.data.Dataset.from_tensor_slices(
            (ambient_testing_fingerprints, ambient_testing_ground_truth)
        ).batch(128).prefetch(tf.data.AUTOTUNE)

        with swap_attribute(model, "reset_metrics", lambda: None):
            ambient_predictions = model.evaluate(ambient_ds, return_dict=True, verbose=0)

        duration_of_ambient_set = (
            data_processor.get_mode_duration("validation_ambient") / 3600.0
        )

        all_true_positives = _to_numpy(ambient_predictions["tp"])
        ambient_fp_raw = _to_numpy(ambient_predictions["fp"])
        all_false_negatives = _to_numpy(ambient_predictions["fn"])

        ambient_false_positives = ambient_fp_raw - test_set_fp

        metrics["auc"] = ambient_predictions["auc"]
        metrics["loss"] = ambient_predictions["loss"]

        recall_at_cutoffs = (
            all_true_positives / (all_true_positives + all_false_negatives)
        )
        faph_at_cutoffs = ambient_false_positives / duration_of_ambient_set

        target_faph_cutoff_probability = 1.0
        for index, cutoff in enumerate(np.linspace(0.0, 1.0, 101)):
            if faph_at_cutoffs[index] == 0:
                target_faph_cutoff_probability = cutoff
                recall_at_no_faph = recall_at_cutoffs[index]
                break

        if faph_at_cutoffs[0] > 2:
            index_of_first_viable = 1
            while faph_at_cutoffs[index_of_first_viable] > 2:
                index_of_first_viable += 1

            x0 = faph_at_cutoffs[index_of_first_viable - 1]
            y0 = recall_at_cutoffs[index_of_first_viable - 1]
            x1 = faph_at_cutoffs[index_of_first_viable]
            y1 = recall_at_cutoffs[index_of_first_viable]

            recall_at_2faph = (y0 * (x1 - 2.0) + y1 * (2.0 - x0)) / (x1 - x0)
        else:
            index_of_first_viable = 0
            recall_at_2faph = recall_at_cutoffs[0]

        x_coordinates = [2.0]
        y_coordinates = [recall_at_2faph]

        for index in range(index_of_first_viable, len(recall_at_cutoffs)):
            if faph_at_cutoffs[index] != x_coordinates[-1]:
                x_coordinates.append(faph_at_cutoffs[index])
                y_coordinates.append(recall_at_cutoffs[index])

        average_viable_recall = (
            np.trapz(np.flip(y_coordinates), np.flip(x_coordinates)) / 2.0
        )

        metrics["recall_at_no_faph"] = recall_at_no_faph
        metrics["cutoff_for_no_faph"] = target_faph_cutoff_probability
        metrics["ambient_false_positives"] = ambient_false_positives[50]
        metrics["ambient_false_positives_per_hour"] = faph_at_cutoffs[50]
        metrics["average_viable_recall"] = average_viable_recall

    return metrics


def train(model, config, data_processor):
    # --- CONFIG PREPARATION ---
    if not (training_steps_list := config.get("training_steps")): training_steps_list = [20000]
    if not (learning_rates_list := config.get("learning_rates")): learning_rates_list = [0.001]
    
    aug_keys = [
        "mix_up_augmentation_prob", "freq_mix_augmentation_prob", 
        "time_mask_max_size", "time_mask_count", 
        "freq_mask_max_size", "freq_mask_count",
        "positive_class_weight", "negative_class_weight"
    ]
    
    augmentation_configs = {}
    training_step_iterations = len(training_steps_list)
    
    def pad_list(lst, length, default=0):
        if not lst: lst = [default]
        while len(lst) < length: lst.append(lst[-1])
        return lst

    key_map = {
        "mix_up_augmentation_prob": "mix_up_prob",
        "freq_mix_augmentation_prob": "freq_mix_prob",
        "time_mask_max_size": "time_mask_max_size",
        "time_mask_count": "time_mask_count",
        "freq_mask_max_size": "freq_mask_max_size",
        "freq_mask_count": "freq_mask_count",
        "positive_class_weight": "positive_class_weight",
        "negative_class_weight": "negative_class_weight"
    }

    pad_list(learning_rates_list, training_step_iterations)
    
    for cfg_key in aug_keys:
        val = config.get(cfg_key)
        if not val:
            if "weight" in cfg_key: val = [1.0]
            elif "prob" in cfg_key: val = [0.0]
            elif "count" in cfg_key: val = [2]
            elif "size" in cfg_key: val = [5]
            else: val = [0]
        pad_list(val, training_step_iterations)
        augmentation_configs[key_map[cfg_key]] = val

    # --- MODEL SETUP ---
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam()
    cutoffs = np.linspace(0.0, 1.0, 101).tolist()

    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.TruePositives(name="tp", thresholds=cutoffs),
        tf.keras.metrics.FalsePositives(name="fp", thresholds=cutoffs),
        tf.keras.metrics.TrueNegatives(name="tn", thresholds=cutoffs),
        tf.keras.metrics.FalseNegatives(name="fn", thresholds=cutoffs),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryCrossentropy(name="loss"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.make_train_function()
    _, model.train_function = tf_decorator.unwrap(model.train_function)

    checkpoint_directory = os.path.join(config["train_dir"], "restore/")
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

    train_writer = tf.summary.create_file_writer(os.path.join(config["summaries_dir"], "train"))
    validation_writer = tf.summary.create_file_writer(os.path.join(config["summaries_dir"], "validation"))

    training_steps_max = np.sum(training_steps_list)
    best_minimization_quantity = 10000
    best_maximization_quantity = 0.0
    best_no_faph_cutoff = 1.0

    # --- START MULTIPROCESSING WORKERS ---
    print("🚀 Starting MULTIPROCESSING data loaders (Using 6 CPU cores)...")
    
    # Use 'spawn' to avoid contaminating workers with TensorFlow GPU context
    ctx = multiprocessing.get_context('spawn')
    data_queue = ctx.Queue(maxsize=30)
    stop_event = ctx.Event()
    processes = []

    # Calculate steps logic once to pass to worker? No, passing lists is fine.
    # Launch 6 workers to saturate CPU
    for _ in range(6):
        p = ctx.Process(
            target=data_module.data_loader_process, 
            args=(config, training_steps_list, learning_rates_list, augmentation_configs, data_queue, stop_event)
        )
        p.start()
        processes.append(p)

    # Calculate step map for MAIN LOOP usage (Learning Rate)
    settings_map = []
    current_sum = 0
    for i in range(len(training_steps_list)):
        current_sum += training_steps_list[i]
        settings_map.append((current_sum, i))

    try:
        idx = 0
        for training_step in range(1, training_steps_max + 1):
            
            # Get data from queue (produced by other cores)
            batch_data = data_queue.get()
            
            # Determine Learning Rate for this step
            if training_step > settings_map[idx][0]:
                if idx < len(settings_map) - 1: idx += 1
            config_idx = settings_map[idx][1]
            learning_rate = learning_rates_list[config_idx]
            
            model.optimizer.learning_rate.assign(learning_rate)

            result = model.train_on_batch(
                batch_data["x"],
                batch_data["y"],
                sample_weight=batch_data["w"],
            )

            print(
                "Validation Batch #{:d}: Accuracy = {:.3f}; Recall = {:.3f}; Precision = {:.3f}; Loss = {:.4f}; Mini-Batch #{:d}".format(
                    (training_step // config["eval_step_interval"] + 1),
                    result[1], result[2], result[3], result[9],
                    (training_step % config["eval_step_interval"]),
                ),
                end="\r",
            )

            is_last_step = training_step == training_steps_max
            if (training_step % config["eval_step_interval"]) == 0 or is_last_step:
                logging.info(
                    "Step #%d: rate %f, accuracy %.2f%%, recall %.2f%%, precision %.2f%%, cross entropy %f",
                    training_step, learning_rate, result[1] * 100, result[2] * 100, result[3] * 100, result[9],
                )

                with train_writer.as_default():
                    tf.summary.scalar("loss", result[9], step=training_step)
                    tf.summary.scalar("accuracy", result[1], step=training_step)
                    tf.summary.scalar("recall", result[2], step=training_step)
                    tf.summary.scalar("precision", result[3], step=training_step)
                    tf.summary.scalar("auc", result[8], step=training_step)
                    train_writer.flush()

                model.save_weights(os.path.join(config["train_dir"], "last_weights.weights.h5"))

                nonstreaming_metrics = validate_nonstreaming(config, data_processor, model, "validation")
                model.reset_metrics()
                
                logging.info(
                    "Step %d (nonstreaming): Validation: recall at no faph = %.3f with cutoff %.2f, accuracy = %.2f%%, recall = %.2f%%, precision = %.2f%%, ambient false positives = %d, estimated false positives per hour = %.5f, loss = %.5f, auc = %.5f, average viable recall = %.9f",
                    training_step,
                    nonstreaming_metrics["recall_at_no_faph"] * 100,
                    nonstreaming_metrics["cutoff_for_no_faph"],
                    nonstreaming_metrics["accuracy"] * 100,
                    nonstreaming_metrics["recall"] * 100,
                    nonstreaming_metrics["precision"] * 100,
                    nonstreaming_metrics["ambient_false_positives"],
                    nonstreaming_metrics["ambient_false_positives_per_hour"],
                    nonstreaming_metrics["loss"],
                    nonstreaming_metrics["auc"],
                    nonstreaming_metrics["average_viable_recall"],
                )

                with validation_writer.as_default():
                    tf.summary.scalar("loss", nonstreaming_metrics["loss"], step=training_step)
                    tf.summary.scalar("accuracy", nonstreaming_metrics["accuracy"], step=training_step)
                    tf.summary.scalar("recall", nonstreaming_metrics["recall"], step=training_step)
                    tf.summary.scalar("precision", nonstreaming_metrics["precision"], step=training_step)
                    tf.summary.scalar("recall_at_no_faph", nonstreaming_metrics["recall_at_no_faph"], step=training_step)
                    tf.summary.scalar("auc", nonstreaming_metrics["auc"], step=training_step)
                    tf.summary.scalar("average_viable_recall", nonstreaming_metrics["average_viable_recall"], step=training_step)
                    validation_writer.flush()

                os.makedirs(os.path.join(config["train_dir"], "train"), exist_ok=True)
                model.save_weights(os.path.join(config["train_dir"], "train", f"{int(best_minimization_quantity * 10000)}_weights_{training_step}.weights.h5"))

                current_minimization_quantity = 0.0
                if config["minimization_metric"] is not None:
                    current_minimization_quantity = nonstreaming_metrics[config["minimization_metric"]]
                current_maximization_quantity = nonstreaming_metrics[config["maximization_metric"]]
                current_no_faph_cutoff = nonstreaming_metrics["cutoff_for_no_faph"]

                improved = False
                if current_minimization_quantity <= config["target_minimization"]:
                    if current_maximization_quantity > best_maximization_quantity or best_minimization_quantity > config["target_minimization"]:
                        improved = True
                elif current_minimization_quantity < best_minimization_quantity:
                    improved = True
                elif current_minimization_quantity == best_minimization_quantity and current_maximization_quantity > best_maximization_quantity:
                    improved = True

                if improved:
                    best_minimization_quantity = current_minimization_quantity
                    best_maximization_quantity = current_maximization_quantity
                    best_no_faph_cutoff = current_no_faph_cutoff
                    model.save_weights(os.path.join(config["train_dir"], "best_weights.weights.h5"))
                    checkpoint.save(file_prefix=checkpoint_prefix)

                logging.info(
                    "So far the best minimization quantity is %.3f with best maximization quantity of %.5f%%; no faph cutoff is %.2f",
                    best_minimization_quantity, (best_maximization_quantity * 100), best_no_faph_cutoff,
                )

    finally:
        stop_event.set()
        for p in processes:
            p.terminate()

    checkpoint.save(file_prefix=checkpoint_prefix)
    model.save_weights(os.path.join(config["train_dir"], "last_weights.weights.h5"))