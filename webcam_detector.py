from ultralytics import YOLO
import cv2, time, collections

MODEL = "runs/detect/asl_yolo/weights/best.pt"  # adjust to your run
CONF   = 0.6
NMS_IOU= 0.5

def main():
    model = YOLO(MODEL)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No camera")

    transcript = []
    last_char = None
    cooldown = 0

    while True:
        ok, frame = cap.read()
        if not ok: break

        res = model.predict(frame, conf=CONF, iou=NMS_IOU, verbose=False)[0]
        text = ""
        if len(res.boxes):
            # take highest-confidence detection
            i = int(res.boxes.conf.argmax())
            cls_id = int(res.boxes.cls[i].item())
            name = model.names[cls_id]
            text = name

            if cooldown > 0: cooldown -= 1
            else:
                if name != last_char:
                    transcript.append(name)
                    last_char = name
                    cooldown = 10  # simple debounce so repeated frames don’t spam

        # draw
        annotated = res.plot()
        cv2.putText(annotated, f"Pred: {text}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Word: {''.join(transcript[-20:])}", (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow("ASL YOLO (Realtime)", annotated)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        if k == ord('c'):
            transcript.clear(); last_char=None; cooldown=0
        if k == ord('b') and transcript:  # backspace
            transcript.pop()

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
