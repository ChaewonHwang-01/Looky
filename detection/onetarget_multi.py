import cv2
import numpy as np
from insightface.app import FaceAnalysis

# -----------------------------------
# 한 명의 타겟에 대해 여러 이미지 제공
# -----------------------------------
target_images = [
    "/Users/kimchaewon/Desktop/capstone/2_cap/target.jpg",
    "/Users/kimchaewon/Desktop/capstone/2_cap/target2.jpg",
]

video_path  = "./highvideo.mp4"
output_path = "./output.mp4"

threshold = 0.45

# -----------------------------------
# InsightFace 모델 준비
# -----------------------------------
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# -----------------------------------
# 여러 reference embedding → 평균 embedding 생성
# -----------------------------------
ref_emb_list = []

for path in target_images:
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"이미지를 열 수 없습니다: {path}")

    faces = app.get(img)
    if len(faces) == 0:
        raise RuntimeError(f"타겟 얼굴을 찾지 못함: {path}")

    ref_emb_list.append(faces[0].normed_embedding)

# 평균 embedding 만들기
ref_emb = np.mean(np.vstack(ref_emb_list), axis=0)
ref_emb = ref_emb / np.linalg.norm(ref_emb)   # L2 정규화

print(f"[INFO] 타겟 임베딩 {len(ref_emb_list)}개 → 평균값 생성 완료")

# -----------------------------------
# 비디오 IO 설정
# -----------------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("비디오를 열 수 없습니다.")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# -----------------------------------
# 프레임 분석
# -----------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        emb = f.normed_embedding

        sim = float(np.dot(emb, ref_emb))  # cosine similarity

        if sim > threshold:
            color, msg = (0, 255, 0), f"FOUND ({sim:.2f})"
        else:
            color, msg = (0, 0, 255), f"{sim:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, msg, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    out.write(frame)
    cv2.imshow("Multi Reference Matching", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("[INFO] 완료")
