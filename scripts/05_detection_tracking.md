# 05_detection_tracking.md — Target Face Matching + Bounding Box (InsightFace)

이 단계는 **최종 HR 영상/프레임**을 입력으로 받아,
InsightFace(ArcFace embedding)를 이용해 **타겟 인물과 유사도(cosine similarity)를 계산**하고,
임계값(threshold) 기준으로 바운딩 박스를 표시한다.

- 유사도 ≥ threshold → **초록 박스(FOUND)**
- 유사도 < threshold → **빨강 박스(Unknown)**

---

## 0) 코드 위치
- `detection/onetarget_multi.py` : 한 명의 타겟에 대해 reference 이미지를 여러 장 사용(평균 embedding)
- `detection/multitarget.py` : 여러 명의 타겟을 각각 등록하여 비교

---

## 1) 입력/출력 예시 경로 (권장)
> 실행 편의를 위해 **절대경로 대신 상대경로**를 권장한다.

### 예시 구조

    data/
    ├─ targets/
    │  ├─ target1.jpg
    │  └─ target2.jpg
    ├─ videos/
    │  └─ highvideo.mp4
    └─ outputs/
       └─ output.mp4

---

## 2) 실행 전 체크(중요)
### (1) 코드 안의 경로를 상대경로로 수정
현재 일부 코드는 `/Users/...`(macOS 절대경로) 형태일 수 있다.
PC에서 재현 가능하게 아래처럼 변경한다.

- 예: `./data/targets/target1.jpg`
- 예: `./data/videos/highvideo.mp4`

### (2) 코덱(H264) 이슈
Windows 환경에서 `H264` 코덱이 안 열릴 수 있다.
그 경우 `mp4v`로 저장되는지 확인하거나, 코드에서 fallback 처리된 버전을 사용한다.

---

## 3) 실행 방법 (Anaconda Prompt)

### (A) Single Target + Multi Reference (추천)
타겟 1명에 대해 reference 이미지 여러 장을 사용해 평균 embedding을 만들고 매칭한다.

```bat
conda activate keep_py310

cd /d <YOUR_REPO_PATH>\Looky

python detection\onetarget_multi.py
