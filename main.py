import gradio as gr
import torch
from PIL import Image
import numpy as np
from transformers import (
    Sam3Processor,
    Sam3Model,
    Sam3TrackerProcessor,
    Sam3TrackerModel,
)
import cv2


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(model_path, device_choice):
    """
    Load SAM3 PCS and PVS models on the requested device.
    Returns models, processors, device string, and a status message.
    """
    device = device_choice or get_default_device()

    try:
        model_pcs = Sam3Model.from_pretrained(model_path).to(device)
        processor_pcs = Sam3Processor.from_pretrained(model_path)

        model_pvs = Sam3TrackerModel.from_pretrained(model_path).to(device)
        processor_pvs = Sam3TrackerProcessor.from_pretrained(model_path)

        status = f" Loaded models from '{model_path}' on device '{device}'."
        return model_pcs, processor_pcs, model_pvs, processor_pvs, device, status
    except Exception as e:
        status = f" Failed to load models from '{model_path}' on '{device}': {e}"
        # Return Nones so downstream can detect missing models
        return None, None, None, None, None, status


def apply_mask_overlay(image, mask, color=(30, 144, 255), alpha=0.5):
    """Overlay a binary mask on an RGB image."""
    if mask is None:
        return image

    overlay = image.copy()
    mask_bool = mask > 0.0
    if not mask_bool.any():
        return overlay

    # Create colored mask
    colored_mask = np.zeros_like(overlay)
    colored_mask[mask_bool] = color

    # Blend
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + colored_mask[mask_bool] * alpha
    ).astype(np.uint8)

    # Draw contours
    try:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    except Exception:
        pass

    return overlay


def segment_text(image, text_prompt, model_pcs, processor_pcs, device):
    """
    Text-prompted segmentation using Sam3Model (PCS).
    """
    if model_pcs is None or processor_pcs is None:
        return (
            None,
            " Models not loaded yet. Please set path/device and click 'Load Models' first.",
        )

    if not text_prompt:
        return None, "Enter a text prompt."

    if image is None:
        return None, "Please upload an image."

    # Split multiple prompts by comma
    prompts = [p.strip() for p in text_prompt.split(",") if p.strip()]

    img_array = np.array(image)
    overlay = img_array.copy()

    total_instances = 0
    category_info = {}

    for idx, prompt in enumerate(prompts):
        # Process inputs with text
        inputs = processor_pcs(images=image, text=prompt, return_tensors="pt").to(
            device
        )

        with torch.no_grad():
            outputs = model_pcs(**inputs)

        # Post-process for instance segmentation
        results = processor_pcs.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=[image.size[::-1]],
        )[0]

        masks = results["masks"]
        num_instances = len(masks)

        if num_instances == 0:
            continue

        total_instances += num_instances
        category_info[prompt] = num_instances

        # Deterministic color per category
        np.random.seed(42 + idx)
        color = np.random.randint(100, 255, 3)

        # Apply masks for this category
        for mask in masks:
            mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
            mask_np = np.squeeze(mask_np)
            while mask_np.ndim > 2:
                mask_np = mask_np[0]

            mask_bool = mask_np > 0

            # 70% color + 30% original
            overlay[mask_bool] = (overlay[mask_bool] * 0.3 + color * 0.7).astype(
                np.uint8
            )

            contours, _ = cv2.findContours(
                mask_np.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, (255, 255, 0), 3)

    if total_instances == 0:
        return image, f" No objects found for: {text_prompt}"

    # Build info message
    info_lines = [f" Found {total_instances} total instance(s):"]
    for prompt, count in category_info.items():
        info_lines.append(f"  • {prompt}: {count}")

    result_image = Image.fromarray(overlay)
    result_info = "\n".join(info_lines)

    return result_image, result_info


def segment_click(image, evt: gr.SelectData, model_pvs, processor_pvs, device):
    """
    Click-based segmentation using Sam3TrackerModel (PVS).
    evt is injected automatically by Gradio; do NOT pass it in inputs.
    """
    if model_pvs is None or processor_pvs is None:
        return (
            None,
            " Models not loaded yet. Please set path/device and click 'Load Models' first.",
        )

    if image is None:
        return None, "Please upload an image."

    # evt.index for Image.select is [x, y]
    x, y = evt.index[0], evt.index[1]

    input_points = [[[[x, y]]]]
    input_labels = [[[1]]]

    inputs = processor_pvs(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model_pvs(**inputs, multimask_output=False)

    masks = processor_pvs.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs.get("original_sizes").tolist(),
        binarize=True,
    )[0]

    if masks is None or masks.numel() == 0:
        return image, f" No object found at ({x}, {y})"

    mask_np = masks[0, 0].cpu().numpy()  # [H, W]

    overlay = np.array(image).copy()
    color_click = np.array([0, 255, 0])

    overlay = apply_mask_overlay(overlay, mask_np, color=color_click)
    cv2.circle(overlay, (x, y), 8, (255, 0, 0), -1)

    return Image.fromarray(overlay), f" Segmented object at ({x}, {y})"


with gr.Blocks() as demo:
    gr.Markdown("# SAM3: Segment Objects with Text or Clicks")

    with gr.Row():
        model_path_input = gr.Textbox(
            value="/path/to/models/sam3",
            label="Model Path",
            placeholder="Path to SAM3 model folder",
        )
        device_default = get_default_device()
        device_dropdown = gr.Dropdown(
            choices=["cpu", "cuda", "mps"],
            value=device_default,
            label="Device",
        )
        load_button = gr.Button("Load Models", variant="primary")

    load_status = gr.Textbox(label="Load Status", interactive=False)

    # States to hold models and device
    model_pcs_state = gr.State()
    processor_pcs_state = gr.State()
    model_pvs_state = gr.State()
    processor_pvs_state = gr.State()
    device_state = gr.State()

    load_button.click(
        load_models,
        inputs=[model_path_input, device_dropdown],
        outputs=[
            model_pcs_state,
            processor_pcs_state,
            model_pvs_state,
            processor_pvs_state,
            device_state,
            load_status,
        ],
    )

    with gr.Tabs():
        with gr.Tab("📝 Text Prompts (Find All)"):
            gr.Markdown(
                "Uses Sam3Model (PCS) to find all instances of a concept given a text prompt."
            )
            with gr.Row():
                with gr.Column():
                    img_text = gr.Image(type="pil", label="Upload Image")
                    prompt = gr.Textbox(
                        label="What to find?",
                        placeholder="e.g., cat, car, person...",
                    )
                    btn_text = gr.Button("🔍 Segment", variant="primary")
                with gr.Column():
                    out_text = gr.Image(type="pil", label="Result")
                    info_text = gr.Textbox(label="Info", lines=3)

            btn_text.click(
                segment_text,
                inputs=[
                    img_text,
                    prompt,
                    model_pcs_state,
                    processor_pcs_state,
                    device_state,
                ],
                outputs=[out_text, info_text],
            )

            prompt.submit(
                segment_text,
                inputs=[
                    img_text,
                    prompt,
                    model_pcs_state,
                    processor_pcs_state,
                    device_state,
                ],
                outputs=[out_text, info_text],
            )

        with gr.Tab("Click to Segment (Specific)"):
            gr.Markdown(
                "Uses Sam3TrackerModel (PVS) to segment a specific object you click."
            )
            with gr.Row():
                with gr.Column():
                    img_click = gr.Image(
                        type="pil",
                        label="Upload Image & Click Object",
                        interactive=True,
                    )
                with gr.Column():
                    out_click = gr.Image(type="pil", label="Result")
                    info_click = gr.Textbox(label="Info", lines=3)

            # Note: evt: gr.SelectData is injected automatically, not passed in inputs
            img_click.select(
                segment_click,
                inputs=[
                    img_click,
                    model_pvs_state,
                    processor_pvs_state,
                    device_state,
                ],
                outputs=[out_click, info_click],
            )

demo.launch()
