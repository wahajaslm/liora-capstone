"""Local inference using the preprocessing of the existing training notebooks."""
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import convnext_tiny
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.categories import CATEGORY_NAMES
from src.text_utils import clean_text, combine_text

PROJECT_ROOT     = Path(__file__).resolve().parents[1]
SVM_MODEL_PATH   = PROJECT_ROOT / "models/svm_tfidf_calibrated.joblib"
TEXT_MODEL_DIR   = PROJECT_ROOT / "models/camembert"
IMAGE_MODEL_PATH = PROJECT_ROOT / "models/outputs/03_7_convnext_tiny_optimized_best.pt"
# train_camembert.py assigns indices in ascending prdtypecode order.
CLASS_CODES      = tuple(sorted(CATEGORY_NAMES))


def smart_white_border_crop(image):

    """Match SmartWhiteBorderCrop in notebook 03_7, including its defaults."""
    # Find the product against a near-white background.
    bbox = image.convert("L").point([255 if i < 245 else 0 for i in range(256)]).getbbox()

    if bbox is None:
        return image
    
    width, height = image.size
    bw            = bbox[2] - bbox[0] 
    bh            = bbox[3] - bbox[1]

    if max(bw / width, bh / height) >= 0.75:
        return image
    # Keep a native-resolution window for small products instead of shrinking them.
    side       = min(max(width, height), max(224, max(bw, bh) + 24))
    cx, cy     = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
    left       = max(0, min(width  - side, cx - side // 2))
    upper      = max(0, min(height - side, cy - side // 2))
    return image.crop((left, upper, left + side, upper + side))


class TextPredictor:
    """Load CamemBERT independently of the image model."""

    def __init__(self):
        # CPU also works on hosts without CUDA and avoids device-specific deployment.
        self.tokenizer  = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR, local_files_only=True)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_DIR, local_files_only=True).eval()

        if self.text_model.config.num_labels != len(CLASS_CODES):
            raise ValueError("CamemBERT must have the project's 27 output classes.")
        
        self.text_error = None

        if any(not torch.isfinite(value).all() for value in self.text_model.state_dict().values()):
            self.text_error = (
                "The saved CamemBERT weights contain NaN or infinite values. "
                "Replace models/camembert with a valid trained export and restart the app."
            )
    @torch.inference_mode()
    def predict(self, title: str, description: str):
        text        = combine_text(clean_text(title), clean_text(description))

        # Invalid exported weights cannot produce a meaningful classification.
        if self.text_error:
            raise ValueError(self.text_error)

        # Use the same cleaning and token limit as the CamemBERT training script.
        encoded = self.tokenizer([text], truncation=True, max_length=128, padding=True, return_tensors="pt")
        scores  = self.text_model(**encoded).logits.softmax(dim=1)[0].numpy()
        if not np.isfinite(scores).all():
            raise ValueError("CamemBERT returned invalid scores. Check the trained model export.")
        return scores


class ImagePredictor:
    """Load ConvNeXt without loading or requiring CamemBERT."""

    def __init__(self):
        # Load only the local checkpoint; no pretrained weights are downloaded.
        checkpoint = torch.load(IMAGE_MODEL_PATH, map_location="cpu", weights_only=True)
        mapping    = {int(code): int(index) for code, index in checkpoint["class_to_index"].items()}

        if set(mapping) != set(CLASS_CODES) or sorted(mapping.values()) != list(range(len(CLASS_CODES))):
            raise ValueError("ConvNeXt class mapping does not match the project's 27 classes.")
        
        self.image_order                = [mapping[code] for code in CLASS_CODES]
        self.image_model                = convnext_tiny(weights=None)
        self.image_model.classifier[2]  = nn.Linear(self.image_model.classifier[2].in_features, len(CLASS_CODES))
        self.image_model.load_state_dict(checkpoint["model_state_dict"])
        self.image_model.eval()
        self.image_transform            = transforms.Compose([
            smart_white_border_crop,
            transforms.Resize(236, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def predict(self, image: Image.Image):
        # Match notebook 03_7: smart crop, resize, center crop and normalization.
        tensor          = self.image_transform(image.convert("RGB")).unsqueeze(0)
        # Average original and horizontally flipped predictions, as in final evaluation.
        scores_original = self.image_model(tensor).softmax(dim=1)
        scores_flipped  = self.image_model(torch.flip(tensor, dims=[3])).softmax(dim=1)
        image_scores    = ((scores_original + scores_flipped) / 2)[0].numpy()
        # Align the checkpoint's output indices with the text model's category order.
        image_scores    = image_scores[self.image_order]
        if not np.isfinite(image_scores).all():
            raise ValueError("ConvNeXt returned invalid scores. Check the trained checkpoint.")
        return image_scores


class SVMPredictor:
    """Load the calibrated TF-IDF + LinearSVC pipeline independently."""

    def __init__(self):
        if not SVM_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing {SVM_MODEL_PATH.name}. Run scripts/export_svm.py to export it."
            )
        self.pipeline = joblib.load(SVM_MODEL_PATH)
        classes = [int(code) for code in self.pipeline.classes_]
        if set(classes) != set(CLASS_CODES):
            raise ValueError("LinearSVC class mapping does not match the project's 27 classes.")
        code_to_index = {code: index for index, code in enumerate(classes)}
        self.class_order = [code_to_index[code] for code in CLASS_CODES]

    def predict(self, title: str, description: str):
        text = combine_text(clean_text(title), clean_text(description))
        scores = self.pipeline.predict_proba([text])[0][self.class_order]
        if not np.isfinite(scores).all():
            raise ValueError("LinearSVC returned invalid scores. Check the exported pipeline.")
        return scores


def fuse_scores(*score_arrays):
    """Equal-weight soft-voting fusion of aligned model probability vectors."""
    valid = [np.asarray(scores, dtype=float) for scores in score_arrays if scores is not None]
    if not valid:
        raise ValueError("At least one score array is required for fusion.")
    return np.mean(valid, axis=0)
