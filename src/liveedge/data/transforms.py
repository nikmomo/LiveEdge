"""Data augmentation transforms for accelerometer data.

This module provides transform classes for augmenting accelerometer data
during training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class BaseTransform(ABC):
    """Abstract base class for data transforms."""

    @abstractmethod
    def __call__(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply the transform.

        Args:
            data: Input data array.

        Returns:
            Transformed data array.
        """
        pass


class Compose(BaseTransform):
    """Compose multiple transforms together.

    Example:
        >>> transform = Compose([
        ...     AddNoise(std=0.05),
        ...     RandomScale(min_scale=0.9, max_scale=1.1),
        ... ])
        >>> augmented = transform(window)
    """

    def __init__(self, transforms: list[BaseTransform]):
        """Initialize the composition.

        Args:
            transforms: List of transforms to apply in order.
        """
        self.transforms = transforms

    def __call__(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply all transforms in sequence.

        Args:
            data: Input data array.

        Returns:
            Transformed data array.
        """
        for transform in self.transforms:
            data = transform(data)
        return data


class RandomApply(BaseTransform):
    """Apply a transform with a given probability.

    Example:
        >>> transform = RandomApply(AddNoise(std=0.1), p=0.5)
    """

    def __init__(self, transform: BaseTransform, p: float = 0.5):
        """Initialize the random apply wrapper.

        Args:
            transform: Transform to apply.
            p: Probability of applying the transform.
        """
        self.transform = transform
        self.p = p

    def __call__(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply the transform with probability p.

        Args:
            data: Input data array.

        Returns:
            Transformed or original data array.
        """
        if np.random.random() < self.p:
            return self.transform(data)
        return data


class AddNoise(BaseTransform):
    """Add Gaussian noise to the data.

    Simulates sensor noise in accelerometer readings.
    """

    def __init__(self, std: float = 0.05, mean: float = 0.0):
        """Initialize the noise transform.

        Args:
            std: Standard deviation of the noise.
            mean: Mean of the noise.
        """
        self.std = std
        self.mean = mean

    def __call__(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Add Gaussian noise to the data.

        Args:
            data: Input data array.

        Returns:
            Noisy data array.
        """
        noise = np.random.normal(self.mean, self.std, data.shape).astype(np.float32)
        return data + noise


class RandomScale(BaseTransform):
    """Randomly scale the data.

    Simulates variations in accelerometer sensitivity or mounting.
    """

    def __init__(
        self,
        min_scale: float = 0.9,
        max_scale: float = 1.1,
        per_axis: bool = False,
    ):
        """Initialize the scale transform.

        Args:
            min_scale: Minimum scale factor.
            max_scale: Maximum scale factor.
            per_axis: If True, apply different scale to each axis.
        """
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.per_axis = per_axis

    def __call__(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply random scaling to the data.

        Args:
            data: Input data array of shape (window_size, n_channels).

        Returns:
            Scaled data array.
        """
        if self.per_axis:
            n_channels = data.shape[-1]
            scale = np.random.uniform(self.min_scale, self.max_scale, n_channels).astype(np.float32)
        else:
            scale = np.random.uniform(self.min_scale, self.max_scale)

        return data * scale


class RandomRotation(BaseTransform):
    """Apply random 3D rotation to accelerometer data.

    Simulates variations in sensor orientation.
    """

    def __init__(self, max_degrees: float = 15.0):
        """Initialize the rotation transform.

        Args:
            max_degrees: Maximum rotation angle in degrees per axis.
        """
        self.max_radians = np.radians(max_degrees)

    def __call__(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply random 3D rotation.

        Args:
            data: Input data array of shape (window_size, 3).

        Returns:
            Rotated data array.
        """
        if data.shape[-1] != 3:
            return data  # Only apply to 3-axis data

        # Random rotation angles
        angles = np.random.uniform(-self.max_radians, self.max_radians, 3)

        # Rotation matrices
        rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angles[0]), -np.sin(angles[0])],
                [0, np.sin(angles[0]), np.cos(angles[0])],
            ]
        )
        ry = np.array(
            [
                [np.cos(angles[1]), 0, np.sin(angles[1])],
                [0, 1, 0],
                [-np.sin(angles[1]), 0, np.cos(angles[1])],
            ]
        )
        rz = np.array(
            [
                [np.cos(angles[2]), -np.sin(angles[2]), 0],
                [np.sin(angles[2]), np.cos(angles[2]), 0],
                [0, 0, 1],
            ]
        )

        # Combined rotation matrix
        rotation = rz @ ry @ rx

        # Apply rotation
        rotated = data @ rotation.T
        return rotated.astype(np.float32)


class TimeWarp(BaseTransform):
    """Apply time warping to the signal.

    Randomly stretches or compresses different parts of the signal.
    """

    def __init__(
        self,
        n_knots: int = 4,
        max_warp: float = 0.2,
    ):
        """Initialize the time warp transform.

        Args:
            n_knots: Number of warp knots.
            max_warp: Maximum warp factor.
        """
        self.n_knots = n_knots
        self.max_warp = max_warp

    def __call__(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply time warping.

        Args:
            data: Input data array of shape (window_size, n_channels).

        Returns:
            Time-warped data array.
        """
        n_samples = data.shape[0]

        # Generate warp path
        orig_steps = np.linspace(0, 1, self.n_knots + 2)
        warp_steps = orig_steps + np.random.uniform(
            -self.max_warp / self.n_knots, self.max_warp / self.n_knots, len(orig_steps)
        )
        warp_steps[0] = 0
        warp_steps[-1] = 1
        warp_steps = np.sort(warp_steps)

        # Interpolate warp path
        from scipy.interpolate import CubicSpline

        warp_fn = CubicSpline(orig_steps, warp_steps)
        time_steps = np.linspace(0, 1, n_samples)
        warped_steps = warp_fn(time_steps)
        warped_steps = np.clip(warped_steps, 0, 1)

        # Resample data
        warped_indices = warped_steps * (n_samples - 1)
        warped_data = np.zeros_like(data)

        for i in range(data.shape[-1]):
            warped_data[:, i] = np.interp(
                np.arange(n_samples),
                warped_indices,
                data[:, i],
            )

        return warped_data.astype(np.float32)


class Permutation(BaseTransform):
    """Randomly permute segments of the signal.

    Useful for testing temporal robustness.
    """

    def __init__(self, n_segments: int = 4, max_segments: int | None = None):
        """Initialize the permutation transform.

        Args:
            n_segments: Number of segments to create.
            max_segments: Maximum segments to permute (None = all).
        """
        self.n_segments = n_segments
        self.max_segments = max_segments or n_segments

    def __call__(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply segment permutation.

        Args:
            data: Input data array of shape (window_size, n_channels).

        Returns:
            Permuted data array.
        """
        n_samples = data.shape[0]
        segment_size = n_samples // self.n_segments

        # Split into segments
        segments = []
        for i in range(self.n_segments):
            start = i * segment_size
            end = start + segment_size if i < self.n_segments - 1 else n_samples
            segments.append(data[start:end])

        # Randomly permute segments
        indices = np.random.permutation(len(segments))
        permuted = np.concatenate([segments[i] for i in indices], axis=0)

        return permuted.astype(np.float32)


class ChannelDropout(BaseTransform):
    """Randomly zero out channels.

    Simulates sensor failure or missing channels.
    """

    def __init__(self, p: float = 0.2):
        """Initialize the channel dropout transform.

        Args:
            p: Probability of dropping each channel.
        """
        self.p = p

    def __call__(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply channel dropout.

        Args:
            data: Input data array of shape (window_size, n_channels).

        Returns:
            Data with some channels zeroed out.
        """
        n_channels = data.shape[-1]
        mask = np.random.random(n_channels) > self.p
        if not mask.any():
            # Ensure at least one channel is kept
            mask[np.random.randint(n_channels)] = True

        result = data.copy()
        result[:, ~mask] = 0
        return result


class MagnitudeWarp(BaseTransform):
    """Apply smooth magnitude warping.

    Multiplies the signal by a smooth random curve.
    """

    def __init__(
        self,
        n_knots: int = 4,
        std: float = 0.2,
    ):
        """Initialize the magnitude warp transform.

        Args:
            n_knots: Number of warp knots.
            std: Standard deviation of warp factors.
        """
        self.n_knots = n_knots
        self.std = std

    def __call__(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply magnitude warping.

        Args:
            data: Input data array of shape (window_size, n_channels).

        Returns:
            Magnitude-warped data array.
        """
        from scipy.interpolate import CubicSpline

        n_samples = data.shape[0]

        # Generate smooth warp curve
        knot_positions = np.linspace(0, n_samples - 1, self.n_knots + 2)
        knot_values = 1 + np.random.normal(0, self.std, len(knot_positions))
        knot_values = np.clip(knot_values, 0.5, 1.5)

        warp_fn = CubicSpline(knot_positions, knot_values)
        warp_curve = warp_fn(np.arange(n_samples))

        # Apply to all channels
        warped = data * warp_curve[:, np.newaxis]
        return warped.astype(np.float32)


def get_default_transforms(
    noise_std: float = 0.05,
    scale_range: tuple[float, float] = (0.9, 1.1),
    rotation_degrees: float = 15.0,
    augment_prob: float = 0.5,
) -> Compose:
    """Get default augmentation transforms for training.

    Args:
        noise_std: Standard deviation of additive noise.
        scale_range: Range of random scaling factors.
        rotation_degrees: Maximum rotation angle in degrees.
        augment_prob: Probability of applying each augmentation.

    Returns:
        Composed transform pipeline.
    """
    return Compose(
        [
            RandomApply(AddNoise(std=noise_std), p=augment_prob),
            RandomApply(RandomScale(min_scale=scale_range[0], max_scale=scale_range[1]), p=augment_prob),
            RandomApply(RandomRotation(max_degrees=rotation_degrees), p=augment_prob),
        ]
    )
