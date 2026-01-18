Synthetic Satellite Inference (Map + Instance Only)
=================================================

This folder contains standalone scripts to generate synthetic satellite RGB imagery using ONLY semantic label maps ("map") and instance polygons ("ins"), without requiring real satellite tiles at inference time. Existing trained checkpoints (e.g. z1 / z2 / z3 experiments) are reused without modifying the original repository code.

Contents
--------
- `infer_map_instance_only.py`: Main batch inference script.
- `validate_basic_metrics.py`: Lightweight validation / sanity metrics on generated imagery.
- `label_instance_dataset.py`: Minimal dataset helper (no real image requirement).

Key Idea
--------
We load the original training options from the checkpoint directory (`opt.pkl`) to reconstruct the generator architecture exactly as trained. Instead of passing a real image through the model, we construct the SPADE semantic tensor from the label map and instance polygons (edges). We then directly call `netG` with a latent vector `z`. For models trained with VAE (`--use_vae`), random latents can still be sampled; if your original training depended on encoding real images for style, expect some style distribution shift. You can try multiple random latents per semantic input to obtain diversity.

Basic Usage
-----------
PowerShell examples (Windows):

```powershell
python .\SPADE\Synt_Sat_Scripts\infer_map_instance_only.py `
  --checkpoints_dir .\SPADE\checkpoints `
  --experiment_name z3_experiment `
  --which_epoch latest `
  --label_dir path\to\label_maps `
  --instance_dir path\to\instance_maps `
  --output_dir .\SPADE\results_map_ins_only `
  --gpu 0 `
  --num_styles 3 `
  --random_latent
```

Then basic validation:

```powershell
python .\SPADE\Synt_Sat_Scripts\validate_basic_metrics.py `
  --generated_dir .\SPADE\results_map_ins_only `
  --label_dir path\to\label_maps
```

Generated images are saved as PNG under the output directory with suffix `_style{k}` when multiple styles are requested.

Suggested Workflow
------------------
1. Ensure your checkpoint directory (e.g. `SPADE/checkpoints/z3_experiment/`) contains `G_{which_epoch}.pth` and `opt.pkl`.
2. Prepare paired `label_dir` and `instance_dir` with matching filenames (e.g. `tile_001.png` in both).
3. Run inference with several `--num_styles` to explore diversity.
4. Run basic validation to inspect color/statistical plausibility.
5. (Optional) Feed generated images into downstream semantic segmentation to assess structural fidelity.

Limitations
-----------
- If the original generator relied heavily on VAE-encoded style from real images, random latent sampling may produce less calibrated colors.
- No discriminator feedback at inference: quality bounded by trained generator distribution.
- Geographic domain shift may cause unrealistic texturing if new semantic configurations differ from training data distribution.

Next Extensions
---------------
- Add script for teacher-student distillation (generator only) using synthetic pseudo-real images.
- Integrate external style code bank from earlier training set for more controlled style transfer.
- Plug in segmentation back-check using a pretrained satellite segmentation model for structural validation.
