import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not read from camera")
    exit()

h, w = frame.shape[:2]
print(f"Frame size: {w}x{h}")

codecs = [
    ("XVID", "output.avi"),
    ("MJPG", "output.avi"),
    ("mp4v", "output.mp4"),
    ("X264", "output.mp4"),
]

for fourcc_str, filename in codecs:
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(filename, fourcc, 30.0, (w, h))
    if writer.isOpened():
        writer.write(frame)
        writer.release()
        print(f"✓ {fourcc_str} -> {filename} WORKS")
    else:
        print(f"✗ {fourcc_str} FAILED")
