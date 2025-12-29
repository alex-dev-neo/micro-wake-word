# coding=utf-8
# Copyright 2024 Kevin Ahrendt.
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

"""Functions and classes for loading/augmenting spectrograms"""

import os
import random
import numpy as np
import time

from absl import logging
from pathlib import Path
from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio.clips import Clips
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.spectrograms import SpectrogramGeneration


def spec_augment(
    spectrogram: np.ndarray,
    time_mask_max_size: int = 0,
    time_mask_count: int = 0,
    freq_mask_max_size: int = 0,
    freq_mask_count: int = 0,
):
    """Applies SpecAugment to the input spectrogram."""
    time_frames = spectrogram.shape[0]
    freq_bins = spectrogram.shape[1]

    augmented_spectrogram = np.copy(spectrogram)

    for i in range(time_mask_count):
        t = int(np.random.uniform(0, time_mask_max_size))
        t0 = random.randint(0, time_frames - t)
        augmented_spectrogram[t0 : t0 + t, :] = 0

    for i in range(freq_mask_count):
        f = int(np.random.uniform(0, freq_mask_max_size))
        f0 = random.randint(0, freq_bins - f)
        augmented_spectrogram[:, f0 : f0 + f] = 0

    return augmented_spectrogram


def fixed_length_spectrogram(
    spectrogram: np.ndarray,
    features_length: int,
    truncation_strategy: str = "random",
    right_cutoff: int = 0,
):
    """Returns a spectrogram with specified length."""
    data_length = spectrogram.shape[0]
    features_offset = 0
    if data_length > features_length:
        if truncation_strategy == "random":
            features_offset = np.random.randint(0, data_length - features_length)
        elif truncation_strategy == "none":
            features_length = data_length
        elif truncation_strategy == "truncate_start":
            features_offset = data_length - features_length
        elif truncation_strategy == "truncate_end":
            features_offset = 0
        elif truncation_strategy == "fixed_right_cutoff":
            features_offset = data_length - features_length - right_cutoff
    else:
        pad_slices = features_length - data_length
        spectrogram = np.pad(
            spectrogram, ((pad_slices, 0), (0, 0)), constant_values=(0, 0)
        )
        features_offset = 0

    return spectrogram[features_offset : (features_offset + features_length)]


class MmapFeatureGenerator(object):
    """A class that handles loading spectrograms from Ragged MMaps."""
    def __init__(
        self,
        path: str,
        label: bool,
        sampling_weight: float,
        penalty_weight: float,
        truncation_strategy: str,
        stride: int,
        step: float,
        fixed_right_cutoffs: list[int] = [0],
    ):
        self.label = float(label)
        self.sampling_weight = sampling_weight
        self.penalty_weight = penalty_weight
        self.truncation_strategy = truncation_strategy
        self.fixed_right_cutoffs = fixed_right_cutoffs
        self.stride = stride
        self.step = step
        self.stats = {}
        self.feature_sets = {}
        self.feature_sets["testing"] = []
        self.feature_sets["training"] = []
        self.feature_sets["validation"] = []
        self.feature_sets["validation_ambient"] = []
        self.feature_sets["testing_ambient"] = []
        self.loaded_features = []

        dirs = ["testing", "training", "validation", "testing_ambient", "validation_ambient"]

        for set_index in dirs:
            duration = 0.0
            count = 0
            search_path_directory = os.path.join(path, set_index)
            search_path = [
                str(i) for i in Path(os.path.abspath(search_path_directory)).glob("**/*_mmap/")
            ]

            for mmap_path in search_path:
                imported_features = RaggedMmap(mmap_path)
                self.loaded_features.append(imported_features)
                feature_index = len(self.loaded_features) - 1

                for i in range(0, len(imported_features)):
                    self.feature_sets[set_index].append(
                        {"loaded_feature_index": feature_index, "subindex": i}
                    )
                    duration += step * imported_features[i].shape[0]
                    count += 1

            random.shuffle(self.feature_sets[set_index])
            self.stats[set_index] = {"spectrogram_count": count, "total_duration": duration}

    def get_mode_duration(self, mode: str):
        return self.stats[mode]["total_duration"]

    def get_mode_size(self, mode):
        return self.stats[mode]["spectrogram_count"]

    def get_random_spectrogram(self, mode: str, features_length: int, truncation_strategy: str):
        right_cutoff = 0
        if truncation_strategy == "default":
            truncation_strategy = self.truncation_strategy
        if truncation_strategy == "fixed_right_cutoff":
            right_cutoff = random.choice(self.fixed_right_cutoffs)

        feature = random.choice(self.feature_sets[mode])
        spectrogram = self.loaded_features[feature["loaded_feature_index"]][feature["subindex"]]
        spectrogram = fixed_length_spectrogram(spectrogram, features_length, truncation_strategy, right_cutoff)

        if np.issubdtype(spectrogram.dtype, np.uint16):
            spectrogram = spectrogram.astype(np.float32) * 0.0390625
        return spectrogram

    def get_feature_generator(self, mode, features_length, truncation_strategy="default"):
        if truncation_strategy == "default":
            truncation_strategy = self.truncation_strategy

        for feature in self.feature_sets[mode]:
            spectrogram = self.loaded_features[feature["loaded_feature_index"]][feature["subindex"]]
            if np.issubdtype(spectrogram.dtype, np.uint16):
                spectrogram = spectrogram.astype(np.float32) * 0.0390625

            if truncation_strategy == "split":
                for feature_start_index in range(
                    0, spectrogram.shape[0] - features_length, int(1000 * self.step * self.stride)
                ):
                    split_spectrogram = spectrogram[feature_start_index : feature_start_index + features_length]
                    yield split_spectrogram
            else:
                for cutoff in self.fixed_right_cutoffs:
                    fixed_spectrogram = fixed_length_spectrogram(spectrogram, features_length, truncation_strategy, cutoff)
                    yield fixed_spectrogram


class ClipsHandlerWrapperGenerator(object):
    def __init__(self, spectrogram_generation: SpectrogramGeneration, label: bool, sampling_weight: float, penalty_weight: float, truncation_strategy: str):
        self.spectrogram_generation = spectrogram_generation
        self.label = label
        self.sampling_weight = sampling_weight
        self.penalty_weight = penalty_weight
        self.truncation_strategy = truncation_strategy
        self.augmented_generator = self.spectrogram_generation.spectrogram_generator(random=True)

    def get_mode_duration(self, mode):
        return 0.0

    def get_mode_size(self, mode):
        if mode == "training":
            return len(self.spectrogram_generation.clips.clips)
        else:
            return 0

    def get_random_spectrogram(self, mode, features_length, truncation_strategy):
        if truncation_strategy == "default":
            truncation_strategy = self.truncation_strategy
        spectrogram = next(self.augmented_generator)
        spectrogram = fixed_length_spectrogram(spectrogram, features_length, truncation_strategy, right_cutoff=0)
        if np.issubdtype(spectrogram.dtype, np.uint16):
            spectrogram = spectrogram.astype(np.float32) * 0.0390625
        return spectrogram

    def get_feature_generator(self, mode, features_length, truncation_strategy="default"):
        for x in []:
            yield x


class FeatureHandler(object):
    def __init__(self, config: dict):
        self.feature_providers = []
        logging.info("Loading and analyzing data sets.")
        for feature_set in config["features"]:
            if feature_set["type"] == "mmap":
                self.feature_providers.append(
                    MmapFeatureGenerator(
                        feature_set["features_dir"],
                        feature_set["truth"],
                        feature_set["sampling_weight"],
                        feature_set["penalty_weight"],
                        feature_set["truncation_strategy"],
                        stride=config["stride"],
                        step=config["window_step_ms"] / 1000.0,
                        fixed_right_cutoffs=feature_set.get("fixed_right_cutoffs", [0]),
                    )
                )
            elif feature_set["type"] == "clips":
                clips_handler = Clips(**feature_set["clips_settings"])
                augmentation_applier = Augmentation(**feature_set["augmentation_settings"])
                spectrogram_generator = SpectrogramGeneration(clips_handler, augmentation_applier, **feature_set["spectrogram_generation_settings"])
                self.feature_providers.append(
                    ClipsHandlerWrapperGenerator(
                        spectrogram_generator,
                        feature_set["truth"],
                        feature_set["sampling_weight"],
                        feature_set["penalty_weight"],
                        feature_set["truncation_strategy"],
                    )
                )
            set_modes = ["training", "validation", "testing", "validation_ambient", "testing_ambient"]
            total_spectrograms = 0
            for set in set_modes:
                total_spectrograms += self.feature_providers[-1].get_mode_size(set)
            if total_spectrograms == 0:
                logging.warning("No spectrograms found in a configured feature set:")
                logging.warning(feature_set)

    def get_mode_duration(self, mode: str):
        sample_duration = 0
        for provider in self.feature_providers:
            sample_duration += provider.get_mode_duration(mode)
        return sample_duration

    def get_mode_size(self, mode: str):
        sample_count = 0
        for provider in self.feature_providers:
            sample_count += provider.get_mode_size(mode)
        return sample_count

    def get_data(
        self,
        mode: str,
        batch_size: int,
        features_length: int,
        truncation_strategy: str = "default",
        augmentation_policy: dict = {
            "freq_mix_prob": 0.0,
            "time_mask_max_size": 0,
            "time_mask_count": 0,
            "freq_mask_max_size": 0,
            "freq_mask_count": 0,
        },
    ):
        if mode == "training":
            sample_count = batch_size
        elif (mode == "validation") or (mode == "testing"):
            sample_count = self.get_mode_size(mode)

        data = []
        labels = []
        weights = []

        if mode == "training":
            random_feature_providers = random.choices(
                [provider for provider in self.feature_providers if provider.get_mode_size("training")],
                [provider.sampling_weight for provider in self.feature_providers if provider.get_mode_size("training")],
                k=sample_count,
            )

            for provider in random_feature_providers:
                spectrogram = provider.get_random_spectrogram("training", features_length, truncation_strategy)
                spectrogram = spec_augment(
                    spectrogram,
                    augmentation_policy["time_mask_max_size"],
                    augmentation_policy["time_mask_count"],
                    augmentation_policy["freq_mask_max_size"],
                    augmentation_policy["freq_mask_count"],
                )
                data.append(spectrogram)
                labels.append(float(provider.label))
                weights.append(float(provider.penalty_weight))
        else:
            for provider in self.feature_providers:
                generator = provider.get_feature_generator(mode, features_length, truncation_strategy)
                for spectrogram in generator:
                    data.append(spectrogram)
                    labels.append(provider.label)
                    weights.append(provider.penalty_weight)

        if truncation_strategy != "none":
            data = np.array(data)
        labels = np.array(labels)
        weights = np.array(weights)

        if truncation_strategy == "none":
            return data, np.array(labels), np.array(weights)

        indices = np.arange(labels.shape[0])
        if mode == "testing" or "validation":
            np.random.shuffle(indices)

        return data[indices], labels[indices], weights[indices]

# --- WORKER FUNCTION FOR MULTIPROCESSING ---
def data_loader_process(config, steps_list, learning_rates, augmentation_configs, q, stop_event):
    """
    Independent process to generate training data.
    Does NOT use TensorFlow, so it avoids GPU memory issues.
    """
    # Create a local data processor instance
    data_processor = FeatureHandler(config)
    
    training_steps_max = np.sum(steps_list)
    settings_map = []
    current_sum = 0
    for i in range(len(steps_list)):
        current_sum += steps_list[i]
        settings_map.append((current_sum, i))

    idx = 0
    # Loop indefinitely or until stop_event
    while not stop_event.is_set():
        # We don't strictly track 'step' here, we just produce batches.
        # But we need 'step' for Learning Rate.
        # Approximation: Main process will consume queue.
        # Let's just generate data. The 'step' in queue is mostly for logging.
        # To be accurate, we'd need shared counter, but let's assume random sampling is uniform.
        # Better: Just use the current settings based on a simple counter
        
        # NOTE: In parallel mode, exact LR schedule per batch is hard to sync perfectly without lock.
        # We will assume 'middle of training' parameters mostly or random?
        # Actually, let's just use the 'current_step' logic if passed, but simpler:
        # Just produce infinite data. The main process handles LR/Step counting.
        
        # We need to know WHICH augmentation to apply.
        # Let's verify 'config_idx'. We can just use the first config for simplicity 
        # OR handle it properly.
        # Simplification: We randomly sample an epoch phase? No.
        
        # Solution: The main process manages the Step/LR. 
        # The worker ONLY provides Raw Batches.
        # Wait, get_data applies augmentation based on policy.
        # We will cycle through policies or pick one? 
        # Let's pick the policy corresponding to the active training phase.
        # Since we have multiple workers, they can just pick 'current' policy if we passed it?
        # No, we can't communicate easily.
        
        # Compromise: We will use the params for the current phase (mostly phase 0 or 1).
        # We will iterate through steps internally to simulate the schedule.
        
        for training_step in range(1, training_steps_max + 1):
            if stop_event.is_set(): return

            if training_step > settings_map[idx][0]:
                if idx < len(settings_map) - 1: idx += 1
            
            config_idx = settings_map[idx][1]
            # LR is handled by main process, we just need augmentation params
            aug_policy = {k: v[config_idx] for k, v in augmentation_configs.items()}
            class_weights = {
                0: augmentation_configs["negative_class_weight"][config_idx],
                1: augmentation_configs["positive_class_weight"][config_idx]
            }

            try:
                # Generate Data
                (x, y, w) = data_processor.get_data(
                    "training",
                    batch_size=config["batch_size"],
                    features_length=config["spectrogram_length"],
                    truncation_strategy="default",
                    augmentation_policy=aug_policy,
                )
                
                # Numpy ops
                y = y.reshape(-1, 1)
                w = w.reshape(-1, 1)
                combined_w = w * np.vectorize(class_weights.get)(y)

                # Push to Queue (Block if full)
                q.put({"x": x, "y": y, "w": combined_w})
                
            except Exception as e:
                logging.error(f"Worker error: {e}")
                time.sleep(1)