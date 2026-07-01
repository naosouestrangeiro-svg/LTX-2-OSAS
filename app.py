#!/usr/bin/env python3
"""
HuggingFace Spaces app.py for LTX-2 Text-to-Video and Image-to-Video Generation
Optimized for ZeroGPU (private HuggingFace Space)

Key ZeroGPU optimizations:
- Models loaded on-demand with @spaces.GPU decorator
- Reduced memory footprint with FP8 quantization
- Efficient batch processing
- Graceful cleanup after each generation
"""

import logging
import os
import tempfile
import traceback
from pathlib import Path

import gradio as gr
import torch

# Import ZeroGPU decorator
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    # Fallback decorator for local testing
    class spaces:
        @staticmethod
        def GPU(duration: int = 0):
            def decorator(func):
                return func
            return decorator

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_model_paths() -> tuple[str, str, str, str]:
    """
    Get model paths from environment variables.
    Set these environment variables in HuggingFace Space settings:
    - LTX2_CHECKPOINT_PATH
    - LTX2_DISTILLED_LORA_PATH
    - LTX2_SPATIAL_UPSAMPLER_PATH
    - LTX2_GEMMA_ROOT
    """
    checkpoint_path = os.getenv("LTX2_CHECKPOINT_PATH")
    distilled_lora_path = os.getenv("LTX2_DISTILLED_LORA_PATH")
    spatial_upsampler_path = os.getenv("LTX2_SPATIAL_UPSAMPLER_PATH")
    gemma_root = os.getenv("LTX2_GEMMA_ROOT")
    
    if not all([checkpoint_path, distilled_lora_path, spatial_upsampler_path, gemma_root]):
        missing = []
        if not checkpoint_path:
            missing.append("LTX2_CHECKPOINT_PATH")
        if not distilled_lora_path:
            missing.append("LTX2_DISTILLED_LORA_PATH")
        if not spatial_upsampler_path:
            missing.append("LTX2_SPATIAL_UPSAMPLER_PATH")
        if not gemma_root:
            missing.append("LTX2_GEMMA_ROOT")
        
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please set these in your HuggingFace Space settings."
        )
    
    return checkpoint_path, distilled_lora_path, spatial_upsampler_path, gemma_root


def validate_paths(
    checkpoint_path: str,
    distilled_lora_path: str,
    spatial_upsampler_path: str,
    gemma_root: str,
) -> tuple[bool, str]:
    """Validate that all required paths exist."""
    paths = {
        "Checkpoint": checkpoint_path,
        "Distilled LoRA": distilled_lora_path,
        "Spatial Upsampler": spatial_upsampler_path,
        "Gemma Root": gemma_root,
    }
    
    for name, path in paths.items():
        if not path or not Path(path).exists():
            return False, f"Missing or invalid path: {name} ({path})"
    
    return True, "All paths valid"


@spaces.GPU(duration=300)  # 5 minutes GPU time per generation
def generate_video(
    prompt: str,
    negative_prompt: str,
    input_image: str | None,
    image_frame_idx: int,
    image_strength: float,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    video_cfg_scale: float,
    video_stg_scale: float,
    audio_cfg_scale: float,
    audio_stg_scale: float,
    checkpoint_path: str,
    distilled_lora_path: str,
    spatial_upsampler_path: str,
    gemma_root: str,
) -> str | None:
    """Generate video from text and optional image conditioning.
    
    This function runs on GPU with ZeroGPU time limit.
    All heavy computation happens here and GPU is released after.
    """
    try:
        logger.info(f"Generating video with prompt: {prompt[:50]}...")
        logger.info(f"Using GPU for generation (up to 300s)")
        
        # Set memory optimization environment variables
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Create distilled LoRA configuration
        distilled_lora = [
            LoraPathStrengthAndSDOps(
                path=distilled_lora_path,
                strength=0.8,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]
        
        # Build FP8 quantization policy for memory efficiency
        logger.info("Building FP8 quantization policy...")
        quantization = build_fp8_cast_policy(checkpoint_path)
        
        # Initialize pipeline with quantization
        logger.info("Loading LTX-2 Two-Stage Pipeline...")
        pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_root,
            loras=[],
            device=device,
            quantization=quantization,
        )
        logger.info("Pipeline loaded successfully!")
        
        # Prepare image conditioning if provided
        images = []
        if input_image is not None and input_image.strip():
            logger.info(f"Using image conditioning: {input_image} at frame {image_frame_idx}")
            images = [
                ImageConditioningInput(
                    path=input_image,
                    frame_idx=image_frame_idx,
                    strength=image_strength,
                    crf=33,  # H.264 compression quality
                )
            ]
        
        # Configure guidance parameters for video and audio
        video_guider_params = MultiModalGuiderParams(
            cfg_scale=video_cfg_scale,
            stg_scale=video_stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[29],
        )
        
        audio_guider_params = MultiModalGuiderParams(
            cfg_scale=audio_cfg_scale,
            stg_scale=audio_stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[29],
        )
        
        # Generate video
        logger.info("Running diffusion stages...")
        video, audio = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            images=images,
            tiling_config=TilingConfig.default(),
            enhance_prompt=False,
            max_batch_size=1,
        )
        
        # Encode video to MP4
        logger.info("Encoding video to MP4...")
        output_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")
        
        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
        
        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )
        
        logger.info(f"Video saved to: {output_path}")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        return output_path
        
    except Exception as e:
        error_msg = f"Error generating video: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        torch.cuda.empty_cache()
        return error_msg


def create_interface():
    """Create the Gradio interface."""
    # Get and validate model paths
    try:
        checkpoint_path, distilled_lora_path, spatial_upsampler_path, gemma_root = get_model_paths()
        is_valid, msg = validate_paths(
            checkpoint_path,
            distilled_lora_path,
            spatial_upsampler_path,
            gemma_root,
        )
        
        if not is_valid:
            logger.error(f"Path validation failed: {msg}")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        checkpoint_path = distilled_lora_path = spatial_upsampler_path = gemma_root = ""
    
    with gr.Blocks(title="LTX-2 Video Generation (ZeroGPU)") as demo:
        gr.Markdown(
            """
            # 🎬 LTX-2 Audio-Video Generation (ZeroGPU)
            Generate videos from text prompts or images with synchronized audio.
            
            **Features:**
            - Text-to-Video generation with high fidelity
            - Image-to-Video conditioning
            - Audio-visual synchronization
            - Two-stage generation (quality + upsampling)
            - Optimized for ZeroGPU (GPU used only during generation)
            
            ⚠️ **Note:** GPU time is limited. Generation will timeout after 5 minutes.
            For optimal results, use lower inference steps (30-40) and smaller frames (100-150).
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Configuration")
                
                prompt = gr.Textbox(
                    label="Text Prompt",
                    placeholder="A serene landscape with mountains in the background...",
                    lines=3,
                    value="A serene landscape with mountains in the background, clear blue sky",
                )
                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="worst quality, blurry, distorted...",
                    lines=2,
                    value="worst quality, low quality, blurry, distorted",
                )
                
                input_image = gr.Image(
                    label="Optional Image Conditioning (Image-to-Video)",
                    type="filepath",
                )
                
                image_frame_idx = gr.Number(
                    label="Image Frame Index",
                    value=0,
                    precision=0,
                )
                
                image_strength = gr.Slider(
                    label="Image Conditioning Strength",
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.1,
                )
                
                seed = gr.Number(
                    label="Seed (for reproducibility)",
                    value=42,
                    precision=0,
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Video Configuration")
                
                height = gr.Slider(
                    label="Video Height (pixels)",
                    minimum=256,
                    maximum=768,
                    value=512,
                    step=64,
                    info="ZeroGPU: use 512 or lower for stability",
                )
                
                width = gr.Slider(
                    label="Video Width (pixels)",
                    minimum=256,
                    maximum=768,
                    value=768,
                    step=64,
                    info="ZeroGPU: use 768 or lower for stability",
                )
                
                num_frames = gr.Slider(
                    label="Number of Frames (8k+1)",
                    minimum=33,
                    maximum=201,
                    value=121,
                    step=8,
                    info="ZeroGPU: use 120 or lower for faster generation",
                )
                
                frame_rate = gr.Slider(
                    label="Frame Rate (fps)",
                    minimum=1,
                    maximum=30,
                    value=25,
                    step=1,
                )
                
                num_inference_steps = gr.Slider(
                    label="Inference Steps (Stage 1)",
                    minimum=20,
                    maximum=50,
                    value=30,
                    step=1,
                    info="ZeroGPU: use 30-40 steps for stability",
                )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Guidance Parameters (Video)")
                
                video_cfg_scale = gr.Slider(
                    label="Video CFG Scale",
                    minimum=1.0,
                    maximum=10.0,
                    value=3.0,
                    step=0.5,
                )
                
                video_stg_scale = gr.Slider(
                    label="Video STG Scale",
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Guidance Parameters (Audio)")
                
                audio_cfg_scale = gr.Slider(
                    label="Audio CFG Scale",
                    minimum=1.0,
                    maximum=10.0,
                    value=7.0,
                    step=0.5,
                )
                
                audio_stg_scale = gr.Slider(
                    label="Audio STG Scale",
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                )
        
        generate_btn = gr.Button("🎬 Generate Video", variant="primary", scale=2)
        
        with gr.Row():
            output_video = gr.Video(label="Generated Video", scale=2)
            output_info = gr.Textbox(
                label="Generation Status/Error",
                interactive=False,
                scale=1,
            )
        
        # Example prompts optimized for ZeroGPU
        gr.Examples(
            examples=[
                [
                    "A serene landscape with mountains in the background, clear blue sky, green meadows",
                    "worst quality, low quality, blurry",
                    None,
                    0,
                    1.0,
                    42,
                    512,
                    768,
                    121,
                    25,
                    30,
                    3.0,
                    1.0,
                    7.0,
                    1.0,
                ],
                [
                    "A person walking through a modern city, skyscrapers in the background, busy streets",
                    "worst quality, low quality, blurry",
                    None,
                    0,
                    1.0,
                    123,
                    512,
                    768,
                    121,
                    25,
                    30,
                    3.0,
                    1.0,
                    7.0,
                    1.0,
                ],
            ],
            inputs=[
                prompt,
                negative_prompt,
                input_image,
                image_frame_idx,
                image_strength,
                seed,
                height,
                width,
                num_frames,
                frame_rate,
                num_inference_steps,
                video_cfg_scale,
                video_stg_scale,
                audio_cfg_scale,
                audio_stg_scale,
            ],
        )
        
        # Generate button click handler
        generate_btn.click(
            fn=generate_video,
            inputs=[
                prompt,
                negative_prompt,
                input_image,
                image_frame_idx,
                image_strength,
                seed,
                height,
                width,
                num_frames,
                frame_rate,
                num_inference_steps,
                video_cfg_scale,
                video_stg_scale,
                audio_cfg_scale,
                audio_stg_scale,
                gr.State(checkpoint_path),
                gr.State(distilled_lora_path),
                gr.State(spatial_upsampler_path),
                gr.State(gemma_root),
            ],
            outputs=[output_video, output_info],
        )
        
        gr.Markdown(
            """
            ### 📋 Usage Guidelines for ZeroGPU
            
            **GPU Time Limit:** 5 minutes per generation
            
            **Recommended Settings for ZeroGPU:**
            - **Resolution:** 512×768 or lower
            - **Frames:** 100-150 (8k+1 format)
            - **Inference Steps:** 30-40 (lower = faster)
            - **Frame Rate:** 25 fps
            
            **Optimization Tips:**
            - Shorter prompts generate faster
            - Fewer inference steps trade quality for speed
            - Reduce frame count to fit within 5-minute GPU window
            - FP8 quantization is enabled for memory efficiency
            
            ### ⚙️ Setup Instructions (for deployment)
            
            Set these environment variables in your HuggingFace Space settings:
            ```
            LTX2_CHECKPOINT_PATH=/path/to/ltx-2.3-22b-dev.safetensors
            LTX2_DISTILLED_LORA_PATH=/path/to/ltx-2.3-22b-distilled-lora-384-1.1.safetensors
            LTX2_SPATIAL_UPSAMPLER_PATH=/path/to/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
            LTX2_GEMMA_ROOT=/path/to/gemma-3-12b-it-qat-q4_0-unquantized
            ```
            
            **Note:** This Space requires a private repository with GPU access (ZeroGPU).
            """
        )
    
    return demo


if __name__ == "__main__":
    logger.info("Starting LTX-2 Video Generation App (ZeroGPU)...")
    logger.info(f"ZeroGPU available: {HAS_SPACES}")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
