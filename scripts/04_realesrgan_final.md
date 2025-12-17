# 04_realesrgan_final.md — Real-ESRGAN x4 (Final Super-Resolution)

본 단계는 CodeFormer를 통해 얼굴 복원이 완료된 프레임을 입력으로 받아,  
**Real-ESRGAN x4를 적용하여 전체 프레임의 전역적인 선명도와 질감을 최종 보정**하는 단계이다.

얼굴 구조는 이전 단계(CodeFormer)에서 이미 확정되었기 때문에,  
본 단계에서는 얼굴 형태를 변경하지 않고 **프레임 전체의 화질 개선만 수행**한다.

---

## 0) 파이프라인 내 위치
Original LR Frames
↓
SwinIR x4 (Global Super-Resolution)
↓
CodeFormer (Face Crop + FaceSR + Merge)
↓
Real-ESRGAN x4 (Final Refinement) ← 현재 단계

---

## 1) 입력 / 출력 경로

### Input (CodeFormer 최종 결과 프레임)
- `C:\Users\hcwon\KEEP\final_results_png`

> CodeFormer 실행 결과 중  
> 얼굴이 원본 프레임에 재합성된 최종 이미지들만 모아둔 디렉터리

### Output (Real-ESRGAN 최종 결과)
- `C:\Users\hcwon\KEEP\video_realesrgan_final`

---

## 2) 환경 준비

Anaconda Prompt에서 다음 명령을 실행한다.

```bat
conda activate keep_py310
cd /d C:\Users\hcwon\Real-ESRGAN

---

## 3) 실행 명령
python inference_realesrgan.py ^
-n RealESRGAN_x4plus ^
-i "C:\Users\hcwon\KEEP\final_results_png" ^
-o "C:\Users\hcwon\KEEP\video_realesrgan_final" ^
--outscale 1 ^
--tile 256 ^
--tile_pad 10 ^
--suffix "" ^
--ext png

