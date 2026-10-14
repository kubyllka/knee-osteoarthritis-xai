"""Shared, immutable settings for the final dataset-wide XAI experiment."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ResearchProfile:
    name: str
    methods: tuple[str, ...]
    metrics: tuple[str, ...]
    faithfulness_runs: int
    faithfulness_subset_size: int
    max_sensitivity_samples: int
    max_sensitivity_normalise: bool
    region_patch_size: int
    region_count: int
    stability_brightness: float
    stability_contrast: float
    integrated_gradients_steps: int
    integrated_gradients_baseline: str
    smoothgrad_samples: int
    smoothgrad_noise_std: float
    resnet_cam_layer: str
    densenet_cam_layer: str

    def metadata(self) -> dict:
        return asdict(self)


FINAL_RESEARCH_PROFILE = ResearchProfile(
    name="final_research",
    methods=("Grad-CAM", "Grad-CAM++", "Eigen-CAM", "Integrated Gradients", "SmoothGrad"),
    metrics=(
        "FaithfulnessCorrelation",
        "Region perturbation score-drop AUC",
        "MaxSensitivity",
        "Stability score",
        "Sparseness",
    ),
    faithfulness_runs=50,
    faithfulness_subset_size=1024,
    max_sensitivity_samples=8,
    max_sensitivity_normalise=True,
    region_patch_size=16,
    region_count=100,
    stability_brightness=1.05,
    stability_contrast=1.05,
    integrated_gradients_steps=16,
    integrated_gradients_baseline="zero_normalised_input",
    smoothgrad_samples=12,
    smoothgrad_noise_std=0.10,
    resnet_cam_layer="layer4[-1]",
    densenet_cam_layer="features.denseblock4",
)
