# 📌 Looky

### Low-Resolution Video Super-Resolution based Missing Person Search System

## 🔍 Qualitative Comparison

| Before (Low-Resolution) | After (Super-Resolution) |
|:--:|:--:|
| ![](assets/videos/before.gif) | ![](assets/videos/after.gif) |

## 1. 프로젝트 개요 (Overview)

**Looky**는  
👉 저해상도 CCTV·영상으로부터 **고해상도(HR) 영상**을 복원하고,  
👉 복원된 영상에서 **특정 실종자를 정확히 탐지·식별**하기 위한  
**AI 기반 영상 복원 + 얼굴 인식 파이프라인**이다.

### 핵심 문제의식
- 실제 실종자 수색 환경에서는 영상 화질이 매우 낮음
- 단순 Super-Resolution만으로는 얼굴 식별에 한계 존재
- 따라서 **복원 → 얼굴 복원 → 최종 식별**의 단계적 접근이 필요

---

## 2. 전체 파이프라인 구조

```text
원본 저해상도 영상
   ↓ (프레임 분할)
SwinIR x4 (1차 HR)
   ↓
CodeFormer (얼굴 crop + 얼굴 복원 + 병합)
   ↓
Real-ESRGAN (전체 프레임 최종 HR)
   ↓
InsightFace 기반 타겟 얼굴 매칭 + Bounding Box

---

## 3. 폴더 구조 설명

```bash
Looky/
│
├─ scripts/                    # 각 단계별 실행/설명 문서
│  ├─ 01_pipeline_overview.md
│  ├─ 02_swinir_x4.md
│  ├─ 03_codeformer_facesr.md
│  ├─ 04_realesrgan_final.md
│  └─ 05_detection_tracking.md
│
├─ detection/                  # 얼굴 인식 및 타겟 매칭 코드
│  ├─ onetarget_multi.py       # 단일 타겟 (reference 여러 장 평균 embedding)
│  └─ multitarget.py           # 다중 타겟 얼굴 인식
│
├─ environment.yml             # Anaconda 실행 환경
├─ .gitignore
└─ README.md

---

## 4. 실행 환경 (Environment)

### 4.1 Anaconda 환경 생성

```bat
conda env create -f environment.yml
conda activate keep_py310

⚠️ GPU 환경 권장
CUDA 사용 가능 시 실행 속도가 크게 향상된다.
