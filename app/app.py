
from pathlib import Path
from io import BytesIO

import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from transformers import SegformerForSemanticSegmentation


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model.pth"

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

IMAGE_SIZE = 384

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CAMO-ViT | Camouflage Segmentation",
    page_icon="🎯",
    layout="wide"
)


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 750;
        margin-bottom: 0;
    }

    .subtitle {
        color: #9CA3AF;
        font-size: 1rem;
        margin-top: 5px;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        margin-top: 20px;
    }

    .info-card {
        padding: 16px;
        border-radius: 10px;
        background-color: rgba(128,128,128,0.10);
        border: 1px solid rgba(128,128,128,0.2);
    }

    .disclaimer {
        font-size: 0.85rem;
        color: #9CA3AF;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model, checkpoint


# =========================================================
# PREPROCESS IMAGE
# =========================================================

def preprocess_image(image):

    original = image.convert("RGB")

    original_array = np.array(original)

    resized = original.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    image_array = (
        np.array(resized).astype(np.float32) / 255.0
    )

    normalized = (image_array - MEAN) / STD

    tensor = torch.from_numpy(normalized)

    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    return original_array, tensor.to(DEVICE)


# =========================================================
# PREDICTION
# =========================================================

@torch.inference_mode()
def generate_prediction(model, tensor, original_shape):

    outputs = model(pixel_values=tensor)

    logits = F.interpolate(
        outputs.logits,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    foreground_probability = probabilities[0, 1]

    prediction = torch.argmax(
        logits,
        dim=1
    )[0]

    prediction = prediction.cpu().numpy().astype(np.uint8)

    probability = foreground_probability.cpu().numpy()

    height, width = original_shape[:2]

    # Resize binary prediction to original dimensions.
    mask_image = Image.fromarray(prediction * 255)

    mask_image = mask_image.resize(
        (width, height),
        Image.Resampling.NEAREST
    )

    mask = np.array(mask_image)

    # Resize probability map to original dimensions.
    probability_image = Image.fromarray(
        (probability * 255).astype(np.uint8)
    )

    probability_image = probability_image.resize(
        (width, height),
        Image.Resampling.BILINEAR
    )

    probability = (
        np.array(probability_image).astype(np.float32) / 255.0
    )

    return mask, probability


# =========================================================
# CREATE VISUALIZATIONS
# =========================================================

def create_outputs(original, mask, probability):

    binary_mask = mask > 127

    # Red overlay.
    red_layer = np.zeros_like(original)
    red_layer[:, :, 0] = 255

    overlay = original.astype(np.float32).copy()

    alpha = 0.45

    overlay[binary_mask] = (
        (1 - alpha) * overlay[binary_mask]
        + alpha * red_layer[binary_mask]
    )

    overlay = np.clip(
        overlay, 0, 255
    ).astype(np.uint8)

    # Probability heatmap.
    colormap = plt.get_cmap("jet")

    heatmap = (
        colormap(probability)[:, :, :3] * 255
    ).astype(np.uint8)

    return (
        Image.fromarray(mask),
        Image.fromarray(overlay),
        Image.fromarray(heatmap)
    )


# =========================================================
# IMAGE DOWNLOAD
# =========================================================

def image_to_bytes(image):

    buffer = BytesIO()

    image.save(buffer, format="PNG")

    return buffer.getvalue()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">CAMO-ViT</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Vision Transformer Framework for Camouflaged Object Segmentation'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("CAMO-ViT")

    st.markdown("### Model Information")

    st.write("**Architecture:** SegFormer-B0")

    st.write("**Task:** Binary Semantic Segmentation")

    st.write("**Dataset:** VIAC (RGB)")

    st.write("**Input resolution:** 384 × 384")

    st.write("**Classes:** Background / Foreground")

    st.divider()

    st.markdown("### Model Performance")

    st.metric("Test Dice", "89.80%")

    st.metric("Test IoU", "81.48%")

    st.caption(
        "Aggregate pixel-level metrics on the "
        "300-image VIAC test set."
    )

    st.divider()

    st.caption(
        "CAMO-ViT is an experimental research prototype."
    )


# =========================================================
# MAIN APPLICATION
# =========================================================

st.markdown(
    "### Camouflage Detection"
)

st.write(
    "Upload an RGB image to generate a predicted "
    "camouflaged-object segmentation mask."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:

        input_image = Image.open(uploaded_file).convert("RGB")

        st.markdown("#### Input Image")

        st.image(
            input_image,
            caption=f"Uploaded image: {uploaded_file.name}",
            width=650
        )

        if st.button(
            "Generate Segmentation",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Loading model and generating segmentation..."
            ):

                model, checkpoint = load_model()

                original, tensor = preprocess_image(
                    input_image
                )

                mask, probability = generate_prediction(
                    model,
                    tensor,
                    original.shape
                )

                mask_image, overlay_image, heatmap_image = (
                    create_outputs(
                        original,
                        mask,
                        probability
                    )
                )

            st.success(
                "Segmentation generated successfully!"
            )

            # ---------------------------------------------
            # RESULTS
            # ---------------------------------------------

            st.divider()

            st.markdown("### Segmentation Results")

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    input_image,
                    caption="Original Image",
                    use_container_width=True
                )

            with col2:

                st.image(
                    mask_image,
                    caption="Predicted Binary Mask",
                    use_container_width=True
                )

            col3, col4 = st.columns(2)

            with col3:

                st.image(
                    overlay_image,
                    caption="Prediction Overlay",
                    use_container_width=True
                )

            with col4:

                st.image(
                    heatmap_image,
                    caption="Foreground Probability Heatmap",
                    use_container_width=True
                )

            # ---------------------------------------------
            # DOWNLOADS
            # ---------------------------------------------

            st.divider()

            st.markdown("### Download Results")

            download1, download2, download3 = st.columns(3)

            with download1:

                st.download_button(
                    "Download Mask",
                    data=image_to_bytes(mask_image),
                    file_name="camo_vit_mask.png",
                    mime="image/png",
                    use_container_width=True
                )

            with download2:

                st.download_button(
                    "Download Overlay",
                    data=image_to_bytes(overlay_image),
                    file_name="camo_vit_overlay.png",
                    mime="image/png",
                    use_container_width=True
                )

            with download3:

                st.download_button(
                    "Download Heatmap",
                    data=image_to_bytes(heatmap_image),
                    file_name="camo_vit_heatmap.png",
                    mime="image/png",
                    use_container_width=True
                )

            # ---------------------------------------------
            # INFORMATION
            # ---------------------------------------------

            st.divider()

            st.markdown("### Inference Information")

            foreground_percentage = (
                np.count_nonzero(mask > 127)
                / mask.size
            ) * 100

            info1, info2, info3 = st.columns(3)

            info1.metric(
                "Image Width",
                f"{original.shape[1]} px"
            )

            info2.metric(
                "Image Height",
                f"{original.shape[0]} px"
            )

            info3.metric(
                "Predicted Foreground",
                f"{foreground_percentage:.2f}%"
            )

            st.caption(
                "Foreground percentage is the proportion of "
                "pixels classified as foreground. It is not "
                "a confidence or accuracy score."
            )

    except Exception as error:

        st.error(
            f"An error occurred: {error}"
        )

else:

    st.info(
        "Upload a JPG, JPEG or PNG image to begin."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="disclaimer">
    CAMO-ViT | Research Prototype<br>
    Predictions are experimental and may contain
    false positives or missed objects, particularly
    on images outside the VIAC dataset distribution.
    </div>
    """,
    unsafe_allow_html=True
)
