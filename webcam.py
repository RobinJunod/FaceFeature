# %%
import cv2, torch, numpy as np
from model import KeypointNet, KeypointNet2, ViTFaceKeypoint, KeypointNet3, KeypointNetM
import time, numpy as np, cv2, torch



MODE = 'vit'
DISPLAY_EDGE = 400      # output height (and width) of *each* panel in pixels
POINT_R = 2             # landmark circle radius

# ──────────────────────────── helpers ────────────────────────────
def center_crop_square(img):
    """Crop longest side so width == height."""
    h, w = img.shape[:2]
    s = min(h, w)
    x0 = (w - s) // 2
    y0 = (h - s) // 2
    return img[y0:y0 + s, x0:x0 + s]

def preprocess(img_square):
    """Square BGR → 96×96 gray tensor (1,1,96,96) & display copy."""
    gray = cv2.cvtColor(img_square, cv2.COLOR_BGR2GRAY)          # (s,s)
    disp96 = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_NEAREST)
    tensor = torch.from_numpy(disp96.astype(np.float32) / 255.0) \
                 .unsqueeze(0).unsqueeze(0)                      # (1,1,96,96)
    return tensor, disp96                                        # tensor for model, uint8 for cv2

# ──────────────────────────── init ───────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if MODE == 'cnn':
    model = KeypointNet().to(device)
    model.load_state_dict(torch.load('weights\cnn_keypointnet2_epoch100.pth', map_location=device))
elif MODE == 'vit':
    # model = ViTFaceKeypoint().to(device)
    # model.load_state_dict(torch.load('weights\\ViTFaceKeypoint_epoch300.pth', map_location=device))
    # model = KeypointNet3().to(device)
    # model.load_state_dict(torch.load('weights\\KeypointNet3_epoch300.pth', map_location=device))
    model = KeypointNetM().to(device)
    model.load_state_dict(torch.load('weights\\KeypointNetM_epoch600.pth', map_location=device))

model.eval()  # set to eval mode
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam.")

# … (all your imports, helper funcs, model loading, VideoCapture …)


# ──────────────────────────── main loop ──────────────────────────
t0 = time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok:
        break

    square   = center_crop_square(frame)                       # BGR, s×s
    inp, lr  = preprocess(square)                              # tensor + 96×96 uint8

    with torch.inference_mode():
        pred = model(inp.to(device))[0].cpu().numpy()          # (30,)

    pts_lr   = (pred.reshape(-1, 2) * 96               ).astype(int)
    pts_hd   = (pred.reshape(-1, 2) * square.shape[0]  ).astype(int)

    # ----- draw landmarks -----
    vis_hd = square.copy()
    vis_lr = cv2.cvtColor(lr, cv2.COLOR_GRAY2BGR)
    for (x96, y96), (xhd, yhd) in zip(pts_lr, pts_hd):
        cv2.circle(vis_lr, (x96, y96), POINT_R, (0, 255, 0), cv2.FILLED)
        cv2.circle(vis_hd, (xhd, yhd), POINT_R, (0, 255, 0), cv2.FILLED)

    # ----- resize panels to identical size -----
    vis_hd = cv2.resize(vis_hd, (DISPLAY_EDGE, DISPLAY_EDGE),
                        interpolation=cv2.INTER_AREA)
    vis_lr = cv2.resize(vis_lr, (DISPLAY_EDGE, DISPLAY_EDGE),
                        interpolation=cv2.INTER_NEAREST)       # blocky on purpose

    # ----- make side-by-side canvas -----
    canvas = np.hstack([vis_hd, vis_lr])

    # ----- FPS overlay -----
    now  = time.perf_counter()
    fps  = 1.0 / (now - t0)
    t0   = now
    cv2.putText(canvas, f'{fps:5.1f} FPS',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2, cv2.LINE_AA)

    # ----- show & exit keys -----
    cv2.imshow('Webcam keypoints: HD VS 96x96', canvas)

    if cv2.waitKey(1) & 0xFF in (27, ord('q')) \
       or cv2.getWindowProperty('Webcam keypoints: HD VS 96x96',  cv2.WND_PROP_VISIBLE) < 1:
        break
cap.release()
cv2.destroyAllWindows()

# %%
