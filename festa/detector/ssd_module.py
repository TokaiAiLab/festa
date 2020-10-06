from typing import List

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SSD_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/ssdpyt_fp32/versions/1/files/nvidia_ssdpyt_fp32_20190225.pt"

utils = torch.hub.load(
    "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd_processing_utils"
)


def load_ssd_model(precision: str = "fp32") -> torch.nn.Module:
    if torch.cuda.is_available():
        device = "cuda"
        model: torch.nn.Module = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd", model_math=precision
        )
    else:
        device = "cpu"
        model = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_ssd",
            model_math=precision,
            pretrained=False,
        )
        checkpoint = torch.hub.load_state_dict_from_url(SSD_URL, map_location=device)
        model.load_state_dict(checkpoint["model"])

    model.to(device)
    return model.eval()


def numpy2tensor(inputs: List[np.ndarray], precision: str = "fp32") -> torch.Tensor:
    """
    Convert to NumPy np.ndarray -> torch.Tensor
    Args:
        inputs: List of np.ndarray [H, W, C] ...[300x300x3]
        precision: default: fp32

    Returns:
        torch.Tensor [N, C, H, W]

    """
    if torch.cuda.is_available():
        tensor: torch.Tensor = utils.prepare_tensor(inputs, precision == "fp16")
    else:
        tensor = torch.stack([torch.tensor(i, dtype=torch.float32) for i in inputs])
        tensor = tensor.permute(0, 3, 1, 2)

    return tensor


class SSDPredictor:
    """
    Examples:
        >>> inputs =  [np.random.randn(300, 300, 3)]  # [H, W, C] ... H == W == 300
        >>> predictor = SSDPredictor()
        >>> _ = predictor.predict(inputs)
        >>> fig, ax = predictor.generate_box()
        >>> plt.plot()
        >>> # OR
        >>> uris = [
        >>>     "http://.....jpg",
        >>>     "http://.....jpg",
        >>>     "http://.....jpg",
        >>> ]
        >>> inputs = [utils.prepare_input(uri) for uri in uris]
        >>> _ = predictor.predict(inputs)
        >>> fig, ax = predictor.generate_box()
        >>> plt.plot()
    """
    def __init__(self, precision: str = "fp32", threshold: float = 0.4):
        self.precision = precision
        self.model: torch.nn.Module = load_ssd_model(precision)
        self.threshold: float = threshold
        self.classes_to_labels = utils.get_coco_object_dictionary()
        self.__best_result_per_input = None
        self.__inputs = None

    def predict(self, inputs: List[np.ndarray], decode_results: bool = True):
        tensor = numpy2tensor(inputs, self.precision)
        with torch.no_grad():
            detections = self.model(tensor)

        if decode_results:
            results_per_input = utils.decode_results(detections)
            best_results_per_input = [
                utils.pick_best(results, self.threshold)
                for results in results_per_input
            ]

            self.__best_result_per_input = best_results_per_input
            self.__inputs = inputs
            return best_results_per_input

        else:
            return detections

    def generate_box(self):
        """
        Examples:
            >>> predictor = SSDPredictor()
            >>> predictor.predict(...)
            >>> predictor.generate_box()
            >>> plt.show()
        """
        if self.__best_result_per_input and self.__inputs:
            for img_idx in range(len(self.__best_result_per_input)):
                fig, ax = plt.subplots(1)
                img = self.__inputs[img_idx] / 2 + 0.5
                ax.imshow(img)

                bboxes, classes, confidences = self.__best_result_per_input[img_idx]

                for idx in range(len(bboxes)):
                    left, bot, right, top = bboxes[idx]
                    x, y, w, h = [
                        val * 300 for val in [left, bot, right - left, top - bot]
                    ]
                    rect = patches.Rectangle(
                        (x, y), w, h, linewidth=1, edgecolor="r", facecolor="none"
                    )
                    ax.add_patch(rect)
                    ax.text(
                        x,
                        y,
                        "{} {:.0f}%".format(
                            self.classes_to_labels[classes[idx] - 1],
                            confidences[idx] * 100,
                        ),
                        bbox=dict(facecolor="white", alpha=0.5),
                    )

                return fig, ax

            self.__best_result_per_input = None
            self.__inputs = None

        else:
            raise RuntimeError("You must always execute `.predict()` before executing this method.")

