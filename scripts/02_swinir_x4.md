# SwinIR x4 Super-Resolution

Environment:
- Conda environment: keep_py310
- Python 3.10

Command:
```bash
python main_test_swinir.py \
  --task real_sr \
  --scale 4 \
  --model_path SwinIR_x4.pth \
  --folder_lq C:\Users\hcwon\KEEP\frames_swin2sr_x4 \
  --folder_gt C:\Users\hcwon\KEEP\frames_swin2sr_x4_faceSR
