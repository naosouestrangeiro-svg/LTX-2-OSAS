# 🎬 LTX-2 HuggingFace Spaces ZeroGPU Deployment

Complete guide for deploying **LTX-2 Text-to-Video and Image-to-Video** to a private HuggingFace Space with ZeroGPU support.

## 📦 What's Included

This deployment package includes:

- **`app.py`** - Full-featured Gradio interface optimized for ZeroGPU
  - Text-to-Video generation
  - Image-to-Video conditioning
  - FP8 quantization for memory efficiency
  - Automatic GPU cleanup
  - 5-minute GPU time limit per generation

- **`requirements.txt`** - All Python dependencies

- **`setup_spaces.py`** - Interactive configuration helper

- **`SPACES_DEPLOYMENT.md`** - Comprehensive deployment guide

- **`.env.example`** - Environment variables template

## 🚀 Quick Start (5 Steps)

### 1. Create Private HuggingFace Space

```
https://huggingface.co/spaces
→ Create new Space
→ Set to PRIVATE (required for ZeroGPU)
→ Choose Python SDK
→ Select ZeroGPU Hardware
```

### 2. Download Model Files

From https://huggingface.co/Lightricks/LTX-2.3:
- `ltx-2.3-22b-dev.safetensors` (Main model)
- `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` (Distilled LoRA)
- `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (Upsampler)

From https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized:
- Download entire Gemma 3 directory

### 3. Upload Files to Space

Either:
- Upload via Space storage/git LFS, or
- Reference from HuggingFace Model Hub

### 4. Set Environment Variables

In your Space **Settings → Variables and secrets**, add:

```
LTX2_CHECKPOINT_PATH = /path/to/ltx-2.3-22b-dev.safetensors
LTX2_DISTILLED_LORA_PATH = /path/to/ltx-2.3-22b-distilled-lora-384-1.1.safetensors
LTX2_SPATIAL_UPSAMPLER_PATH = /path/to/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
LTX2_GEMMA_ROOT = /path/to/gemma-3-12b-it-qat-q4_0-unquantized
```

**Important:** Use "Add secret" not "Add variable" to keep paths hidden from git!

### 5. Push Files to Space

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/your-space-name
cd your-space-name

# Copy from this repo
cp app.py requirements.txt setup_spaces.py .env.example ./

git add .
git commit -m "Add LTX-2 ZeroGPU deployment"
git push
```

Space will auto-deploy and you're done! 🎉

## 💡 Key Features

### ZeroGPU Optimizations
- ⏱️ **5-minute GPU allocation** - Per generation time limit
- 💾 **FP8 Quantization** - 50% memory reduction
- 🧹 **Auto GPU cleanup** - Prevents memory leaks
- ⚡ **Expandable CUDA** - Better memory allocation

### Video Generation Capabilities
- 🎥 **Text-to-Video** - From prompts alone
- 🖼️ **Image-to-Video** - Condition on images
- 🎵 **Audio-Visual Sync** - Synchronized audio generation
- 🔄 **Two-Stage Generation** - Quality + upsampling

### Recommended Settings (for 5-minute GPU limit)
```
Resolution: 512×768
Frames: 100-150 (8k+1 format)
Inference Steps: 30-40
Frame Rate: 25 fps
Typical Time: 2-3 minutes
```

## 📚 Documentation

- **[SPACES_DEPLOYMENT.md](SPACES_DEPLOYMENT.md)** - Full step-by-step guide
- **[.env.example](.env.example)** - Environment variables reference
- **[setup_spaces.py](setup_spaces.py)** - Interactive setup helper
- **[LTX-2 Model Docs](https://huggingface.co/Lightricks/LTX-2.3)** - Model details

## 🔧 Usage

Once deployed, the Gradio interface provides:

### Input Section
- **Prompt** - Detailed video description
- **Negative Prompt** - What to avoid
- **Optional Image** - For image-to-video conditioning
- **Seed** - For reproducibility

### Video Settings
- **Resolution** - 256-768px (height/width)
- **Frames** - 33-201 (8k+1 format)
- **Frame Rate** - 1-30 fps
- **Inference Steps** - 20-50 (lower = faster)

### Guidance Parameters
- **Video CFG Scale** - Text adherence (1-10)
- **Video STG Scale** - Temporal coherence (0-2)
- **Audio CFG Scale** - Audio prompt adherence (1-10)
- **Audio STG Scale** - Audio coherence (0-2)

## 🐛 Troubleshooting

### Missing Environment Variables
```
Error: Missing required environment variables: LTX2_CHECKPOINT_PATH, ...
```
**Solution:** Ensure ALL 4 variables are set in Space settings as secrets (not variables).

### GPU Timeout
```
Error: GPU time limit exceeded (5 minutes)
```
**Solution:** Reduce settings:
- Lower inference steps (20-30)
- Reduce frames (80-120)
- Smaller resolution (384×512)

### Out of Memory
```
Error: CUDA out of memory
```
**Solution:** FP8 quantization is enabled. If still OOM:
- Reduce resolution
- Fewer frames
- Lower batch size (already 1)

### Model Files Not Found
```
Error: Path not found: /path/to/model
```
**Solution:** Verify paths:
```bash
ls /path/to/ltx-2.3-22b-dev.safetensors
ls /path/to/gemma-3-12b-it-qat-q4_0-unquantized
```

## 📊 Performance Metrics

| Setting | Time | Quality | GPU Memory |
|---------|------|---------|-----------|
| 512×768, 121 frames, 30 steps | ~2-3 min | High | ~24GB |
| 384×512, 100 frames, 25 steps | ~1.5-2 min | Good | ~20GB |
| 320×480, 80 frames, 20 steps | ~1 min | Fair | ~16GB |

## 💰 Cost & Credits

- **ZeroGPU Free Tier:** ~50-100 GPU hours/month
- **Cost per Generation:** ~2-3 minutes GPU time
- **Typical Usage:** 10 videos/day = 12+ hours/month

## 🔐 Security

- ✅ **Private Space** - Only you can access
- ✅ **Secrets** - Model paths hidden from git
- ✅ **No API** - Unlike public spaces, no public inference
- ✅ **Rate Limited** - By HuggingFace infrastructure

## 📖 Example Prompts

### Text-to-Video
```
"A serene landscape with mountains in the background, 
clear blue sky with white clouds, green meadows in the 
foreground, gentle breeze moving grass, birds flying"
```

### Image-to-Video
```
Upload an image at frame 0
Prompt: "The person in the image starts walking 
through a city street, looking around, buildings 
passing by in the background"
```

## 🔗 Resources

- **LTX-2 Model:** https://huggingface.co/Lightricks/LTX-2.3
- **Gemma Encoder:** https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
- **HF Spaces Docs:** https://huggingface.co/docs/hub/spaces
- **ZeroGPU Guide:** https://huggingface.co/docs/hub/spaces-gpu
- **Original Repo:** https://github.com/Lightricks/LTX-2

## ✅ Checklist Before Deployment

- [ ] HuggingFace account with GPU credits
- [ ] Private Space created with ZeroGPU hardware
- [ ] All 4 model files downloaded
- [ ] Environment variables set in Space settings (as secrets)
- [ ] Files pushed to Space repository
- [ ] Space logs show successful deployment

## 🎉 Support

For issues:
1. Check **Space Logs** tab for errors
2. Review **SPACES_DEPLOYMENT.md** troubleshooting
3. Verify environment variables in Space settings
4. Ensure model files exist at specified paths

---

**Ready to deploy?** Start with the [SPACES_DEPLOYMENT.md](SPACES_DEPLOYMENT.md) guide! 🚀
