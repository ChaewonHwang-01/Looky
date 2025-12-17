import cv2
import numpy as np
from insightface.app import FaceAnalysis

# -----------------------------------
# 여러 명의 타겟 얼굴 경로 설정
# -----------------------------------
target_paths = [
    ("./target1.jpg", "Person1"),
    ("./target2.jpg", "Person2"),
]

video_path  = "./highvideo.mp4"
output_path = "./output.mp4"

threshold = 0.45  # 유사도 기준

# -----------------------------------
# InsightFace: face detector + ArcFace embedding
# -----------------------------------
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# -----------------------------------
# 여러 타겟 얼굴 임베딩 로드
# -----------------------------------
ref_embeddings = []   # (label, embedding)

for path, label in target_paths:
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"타겟 이미지를 열 수 없습니다: {path}")

    faces = app.get(img)
    if len(faces) == 0:
        raise RuntimeError(f"타겟 얼굴을 찾지 못했습니다: {path}")

    emb = faces[0].normed_embedding
    ref_embeddings.append((label, emb))

print(f"[INFO] 로드된 타겟 수: {len(ref_embeddings)}명")

# -----------------------------------
# 비디오 입출력 준비
# -----------------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("비디오를 열 수 없습니다.")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

fourcc = cv2.VideoWriter_fourcc(*"H264")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
if not out.isOpened():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

print("[INFO] 영상 분석 시작... (q로 종료)")

# -----------------------------------
# 프레임 루프
# -----------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    if len(faces) == 0:
        cv2.putText(frame, "No face", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    else:
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            emb = f.normed_embedding

            best_sim = -1
            best_label = None

            # ------------------------------
            # 모든 타겟과 비교
            # ------------------------------
            for label, ref_emb in ref_embeddings:
                sim = float(np.dot(emb, ref_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_label = label

            # ------------------------------
            # threshold 체크 후 표시
            # ------------------------------
            if best_sim >= threshold:
                color = (0, 255, 0)
                msg = f"{best_label} ({best_sim:.2f})"
            else:
                color = (0, 0, 255)
                msg = f"Unknown ({best_sim:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, msg, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    out.write(frame)
    cv2.imshow("Multi-target Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("[INFO] 완료")
