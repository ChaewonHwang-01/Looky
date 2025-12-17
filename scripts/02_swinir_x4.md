# SwinIR x4 Super-Resolution

Environment:
- Conda environment: keep_py310
- Python 3.10

Command:
```bash
python main_test_swinir_tile.py ^
  --scale 4 ^
  --model_path "C:\Users\hcwon\mmagic\Looky\SwinIR\model_zoo\003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth" ^
  --folder_lq "C:\Users\hcwon\KEEP\video_lr_frame_min" ^
  --folder_gt "C:\Users\hcwon\KEEP\video_swin2sr_x4" ^
  --tile 128 ^
  --tile_overlap 32
