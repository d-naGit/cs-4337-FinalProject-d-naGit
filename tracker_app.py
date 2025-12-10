import cv2
import numpy as np
import argparse

def create_tracker(tracker_type):
    """
    Helper to create an OpenCV tracker (handles legacy vs non-legacy namespaces).
    """
    tracker_type = tracker_type.lower()
    legacy = getattr(cv2, "legacy", None)

    def _create(name):
        # Try cv2.legacy.TrackerX_create first (for newer OpenCV)
        if legacy is not None and hasattr(legacy, name):
            return getattr(legacy, name)()
        # Fallback: cv2.TrackerX_create (older style)
        if hasattr(cv2, name):
            return getattr(cv2, name)()
        raise AttributeError(f"Tracker type '{name}' is not available in your OpenCV build.")

    if tracker_type == "csrt":
        return _create("TrackerCSRT_create")
    elif tracker_type == "kcf":
        return _create("TrackerKCF_create")
    elif tracker_type == "mosse":
        return _create("TrackerMOSSE_create")
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")

def run_tracking(video_path, methods):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    ok, first_frame = cap.read()
    if not ok:
        print("[ERROR] Could not read first frame from video.")
        return

    # --- User ROI selection using cv2.selectROI ---
    print("[INFO] A window called 'Select ROI' should appear.")
    print("       Use the mouse to drag a box around your object.")
    print("       Then press ENTER or SPACE to confirm, or 'c' to cancel.")
    init_bbox = cv2.selectROI("Select ROI", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")  # close ROI window

    x, y, w, h = init_bbox
    if w == 0 or h == 0:
        print("[ERROR] Empty ROI selected. Exiting.")
        return

    trackers = {}
    lost = {}
    survival = {}
    use_template = "template" in methods

    # --- Template matching initialization ---
    if use_template:
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        template = first_gray[y:y + h, x:x + w].copy()
        trackers["template"] = None  # pseudo entry for consistent handling
        lost["template"] = False
        survival["template"] = 0

    # --- OpenCV tracker initialization (CSRT/KCF/MOSSE) ---
    for m in methods:
        if m == "template":
            continue
        try:
            t = create_tracker(m)
            t.init(first_frame, (x, y, w, h))
            trackers[m] = t
            lost[m] = False
            survival[m] = 0
            print(f"[INFO] Initialized tracker: {m}")
        except Exception as e:
            print(f"[WARN] Could not create tracker '{m}': {e}")

    if not trackers:
        print("[ERROR] No valid trackers initialized. Exiting.")
        return

    total_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total_frames += 1

        display_frame = frame.copy()

        # --- Template Matching Tracker ---
        if use_template and not lost["template"]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            # Simple threshold to say "we've lost the object"
            threshold = 0.5
            if max_val < threshold:
                lost["template"] = True
            else:
                tx, ty = max_loc
                tw, th = template.shape[1], template.shape[0]
                survival["template"] += 1
                cv2.rectangle(display_frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2)
                cv2.putText(display_frame, f"template ({max_val:.2f})",
                            (tx, max(ty - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1, cv2.LINE_AA)

        # --- Built-in trackers: CSRT/KCF/MOSSE ---
        for name, tracker in trackers.items():
            if name == "template":
                continue  # template already handled above
            if lost.get(name, True):
                continue

            ok, bbox = tracker.update(frame)
            if not ok:
                lost[name] = True
                continue

            bx, by, bw, bh = [int(v) for v in bbox]
            if bw <= 0 or bh <= 0:
                lost[name] = True
                continue

            survival[name] += 1

            # Give each tracker a distinct color
            if name == "csrt":
                color = (0, 0, 255)   # red-ish
            elif name == "kcf":
                color = (255, 0, 0)   # blue-ish
            elif name == "mosse":
                color = (0, 255, 255) # yellow-ish
            else:
                color = (255, 255, 255)

            cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), color, 2)
            cv2.putText(display_frame, name,
                        (bx, max(by - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1, cv2.LINE_AA)

        # --- Show tracking result ---
        cv2.imshow("Tracking comparison", display_frame)
        key = cv2.waitKey(30) & 0xFF

        # Press 'q' or ESC to quit early
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- Print survival summary ---
    print("\n=== Tracking survival summary ===")
    print(f"Total frames processed: {total_frames}")
    for name in survival:
        print(f"{name:9s}: survived {survival[name]} / {total_frames} frames")

def main():
    parser = argparse.ArgumentParser(
        description="Object tracking comparison: template matching vs OpenCV trackers"
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to input video file (e.g., dataset video)"
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["template", "csrt", "kcf", "mosse"],
        choices=["template", "csrt", "kcf", "mosse"],
        help="Tracking methods to run (default: all)"
    )
    args = parser.parse_args()
    run_tracking(args.video, args.methods)

if __name__ == "__main__":
    main()