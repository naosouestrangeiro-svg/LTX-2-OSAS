# HuggingFace Spaces ZeroGPU Deployment Guide for LTX-2

This guide walks you through deploying the LTX-2 video generation model to a **private HuggingFace Space** with **ZeroGPU** support.

## 📋 Prerequisites

1. **HuggingFace Account** with GPU credits
2. **Private HuggingFace Space** (ZeroGPU requires private spaces)
3. **Downloaded Model Files** (~60GB total):
   - `ltx-2.3-22b-dev.safetensors` (Main model)
   - `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` (Distilled LoRA)
   - `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (Upsampler)
   - `gemma-3-12b-it-qat-q4_0-unquantized/` (Text encoder)

## 🚀 Step-by-Step Deployment

### 1. Create a Private HuggingFace Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Choose:
   - **Owner:** Your account
   - **Space name:** `ltx-2-video-gen` (or your choice)
   - **License:** MIT or Apache 2.0
   - **Private:** ✅ Check this box (ZeroGPU requires private spaces)
   - **Space SDK:** Python
   - **Space hardware:** ZeroGPU

### 2. Clone Your Space

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/ltx-2-video-gen
cd ltx-2-video-gen
```

### 3. Set Up the Repository

```bash
# Copy files from LTX-2-OSAS
git clone https://github.com/naosouestrangeiro-svg/LTX-2-OSAS.git
cd ltx-2-video-gen

# Copy the main app
cp ../LTX-2-OSAS/app.py .

# Copy requirements
cp ../LTX-2-OSAS/requirements.txt .

# Copy setup helper
cp ../LTX-2-OSAS/setup_spaces.py .
```

### 4. Configure Environment Variables

#### Option A: Using the Setup Helper (Recommended)

```bash
python setup_spaces.py
```

This will guide you through entering your model paths interactively.

#### Option B: Manual Configuration

1. Go to your Space Settings → **Variables and secrets**
2. Click **"Add secret"** for each variable (secrets are not shown in git):

```
LTX2_CHECKPOINT_PATH=/path/to/ltx-2.3-22b-dev.safetensors
LTX2_DISTILLED_LORA_PATH=/path/to/ltx-2.3-22b-distilled-lora-384-1.1.safetensors
LTX2_SPATIAL_UPSAMPLER_PATH=/path/to/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
LTX2_GEMMA_ROOT=/path/to/gemma-3-12b-it-qat-q4_0-unquantized
```

### 5. Upload Model Files

HuggingFace Spaces have limited initial storage. You have several options:

#### Option A: Use HuggingFace Model Hub (Recommended)

1. Create a new **Private Model Repository** on HuggingFace
2. Upload your model files there
3. In Space settings, set the paths to `/mnt/data/model_name` (if using persistent storage)
4. Or reference via `huggingface_hub` to download on first run

#### Option B: Use Space Storage (if space allows)

1. In your Space folder, create directories for models:
   ```bash
   mkdir -p models/checkpoint models/lora models/upsampler models/gemma
   ```

2. Upload files via git LFS:
   ```bash
   git lfs install
   git lfs track "models/**/*.safetensors"
   git add .gitattributes
   git add models/
   git commit -m "Add model files"
   git push
   ```

#### Option C: Download from HuggingFace Hub on Runtime

Add to `app.py` before loading models:

```python
from huggingface_hub import hf_hub_download

checkpoint_path = hf_hub_download(
    repo_id="Lightricks/LTX-2.3",
    filename="ltx-2.3-22b-dev.safetensors",
    cache_dir="/tmp/models"
)
```

### 6. Push to Space

```bash
git add app.py requirements.txt setup_spaces.py
git commit -m "Initial LTX-2 ZeroGPU deployment"
git push
```

The Space will automatically:
1. Install requirements from `requirements.txt`
2. Run `app.py`
3. Load the Gradio interface

## 🎯 Usage

Once deployed:

1. Open your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/ltx-2-video-gen`
2. Fill in the form:
   - **Prompt:** Describe the video you want to generate
   - **Optional Image:** Provide an image for image-to-video
   - **Video Settings:** Adjust resolution, duration, frames
   - **Guidance:** Fine-tune CFG and STG scales
3. Click **"Generate Video"**
4. Wait for GPU allocation (should be instant if credits available)
5. Download the generated MP4

## ⚙️ ZeroGPU Configuration Tips

### GPU Time Management

The default is **300 seconds (5 minutes)** per generation. To adjust:

```python
@spaces.GPU(duration=300)  # seconds
def generate_video(...):
    ...
```

### Memory Optimization

The app includes:
- ✅ **FP8 Quantization** - Reduces model size by ~50%
- ✅ **Expandable CUDA Segments** - Better memory allocation
- ✅ **Automatic GPU Cache Cleanup** - Prevents memory leaks

### Recommended Settings for ZeroGPU

For fastest generation within 5-minute limit:

```
Resolution: 512×768
Frames: 100-150 (8k+1 format)
Inference Steps: 30-40
Frame Rate: 25 fps
```

This typically completes in 2-3 minutes.

## 🐛 Troubleshooting

### "Missing environment variables" error

**Solution:** Ensure all 4 variables are set in Space settings:
- LTX2_CHECKPOINT_PATH
- LTX2_DISTILLED_LORA_PATH
- LTX2_SPATIAL_UPSAMPLER_PATH
- LTX2_GEMMA_ROOT

### GPU timeout (5 minutes exceeded)

**Solution:** Reduce settings:
- Lower inference steps (20-30)
- Reduce frame count (80-120)
- Lower resolution (384×512)

### Out of Memory (OOM) errors

**Solution:** The app has FP8 quantization enabled. If still OOM:
- Reduce batch size (already set to 1)
- Lower resolution
- Use fewer frames

### Models not loading

**Solution:** Verify paths are correct:
```bash
ls /path/to/ltx-2.3-22b-dev.safetensors
ls /path/to/gemma-3-12b-it-qat-q4_0-unquantized
```

## 📊 Monitoring

Monitor your Space in the **"Logs"** tab:
- Shows GPU allocation/deallocation
- Inference progress
- Memory usage
- Any errors or warnings

## 💰 Cost Estimation

With ZeroGPU:
- **Free tier:** Limited hours per month
- **Pro tier:** ~$9-12 per month for more GPU hours
- **Each generation:** ~2-3 minutes GPU time

Example costs:
- 10 videos/day × 2.5 min = 25 min/day = ~12 hours/month
- Free tier usually gives 50-100 hours/month

## 🔐 Security Notes

- **Private Space:** Only you can access
- **Secrets:** Model paths are hidden from git
- **No public inference API:** Unlike public spaces
- **Rate limited:** By HuggingFace infrastructure

## 📚 Additional Resources

- **LTX-2 Model Card:** https://huggingface.co/Lightricks/LTX-2.3
- **Gemma Model:** https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
- **HuggingFace Spaces Docs:** https://huggingface.co/docs/hub/spaces-config-reference
- **ZeroGPU Guide:** https://huggingface.co/docs/hub/spaces-gpu

## 🎉 Done!

Your LTX-2 ZeroGPU Space is now live! Share the URL with collaborators or keep it private.

---

**Need Help?**
- Check Space logs for errors
- Review this guide's troubleshooting section
- Open an issue on the LTX-2-OSAS repository
