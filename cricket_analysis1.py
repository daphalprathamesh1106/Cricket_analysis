#!/usr/bin/env python3
"""
cricket_full_segment_analysis.py

Outputs (per video):
  Outputs/<video_name>/
    batting_segments/
    bowling_segments/
    follow_through_segments/
    fielding_segments/

Each segment folder contains:
  - segment_<i>_keypoints.csv
  - segment_<i>_clip.mp4
  - segment_<i>_skeleton.mp4
  - segment_<i>_summary.txt
"""
import os
import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tempfile
import shutil
import math

# ========== CONFIG ==========
MIN_SEGMENT_FRAMES = 6        # minimum frames to consider a segment valid
FPS_OUTPUT = 25               # fallback fps for exported clips
POSE_LANDMARK_COUNT = 33

# MediaPipe landmark indices (for readability)
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Put your local video paths here
video_paths = [
    "/Users/prathameshdaphal/Desktop/Job/Tests/Sportler/Video-1.mp4",
    "/Users/prathameshdaphal/Desktop/Job/Tests/Sportler/Video-2.mp4",
    "/Users/prathameshdaphal/Desktop/Job/Tests/Sportler/Video-3.mp4",
    "/Users/prathameshdaphal/Desktop/Job/Tests/Sportler/Video-4.mp4",
    "/Users/prathameshdaphal/Desktop/Job/Tests/Sportler/Video-5.mp4",
]

OUTPUT_ROOT = "/Users/prathameshdaphal/Desktop/Job/Tests/Sportler/Outputs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Initialize MediaPipe once
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=False,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

# ========== helpers ==========
def calculate_angle(a, b, c):
    """Return angle ABC in degrees. a,b,c are 2D iterables [x,y]."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return float("nan")
    cosine = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

def euclidean(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.linalg.norm(a - b))

# ========== extraction ==========
def extract_all_keypoints(video_path):
    """Process every frame and return df with keypoints and fps."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or math.isnan(fps) or fps <= 0:
        fps = FPS_OUTPUT

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose_detector.process(image_rgb)
        row = {"frame": frame_idx}
        if res.pose_landmarks:
            for i, lm in enumerate(res.pose_landmarks.landmark):
                row[f"x_{i}"] = float(lm.x)
                row[f"y_{i}"] = float(lm.y)
                row[f"z_{i}"] = float(lm.z)
                row[f"v_{i}"] = float(lm.visibility)
        else:
            for i in range(POSE_LANDMARK_COUNT):
                row[f"x_{i}"] = np.nan
                row[f"y_{i}"] = np.nan
                row[f"z_{i}"] = np.nan
                row[f"v_{i}"] = np.nan
        frames.append(row)
        frame_idx += 1
    cap.release()
    df = pd.DataFrame(frames)
    return df, fps

# ========== frame classifier (heuristic) ==========
def classify_frame(row):
    """
    Heuristic rules:
      - bowling: right wrist above head (rw_y < nose_y)
      - fielding: knees relatively low (knee_y significantly > shoulder_y)
      - batting: ankles roughly level (standing) and wrists near shoulders
      - otherwise follow_through
    """
    def has(idx):
        return not pd.isna(row.get(f"x_{idx}", np.nan)) and not pd.isna(row.get(f"y_{idx}", np.nan))

    if not has(NOSE):
        return "none"

    nose_y = row.get(f"y_{NOSE}", np.nan)
    rw_y = row.get(f"y_{RIGHT_WRIST}", np.nan)
    lw_y = row.get(f"y_{LEFT_WRIST}", np.nan)
    l_knee_y = row.get(f"y_{LEFT_KNEE}", np.nan)
    r_knee_y = row.get(f"y_{RIGHT_KNEE}", np.nan)
    l_sh_y = row.get(f"y_{LEFT_SHOULDER}", np.nan)
    r_sh_y = row.get(f"y_{RIGHT_SHOULDER}", np.nan)
    l_ankle_y = row.get(f"y_{LEFT_ANKLE}", np.nan)
    r_ankle_y = row.get(f"y_{RIGHT_ANKLE}", np.nan)

    # bowling -> wrist above head
    if (not pd.isna(rw_y) and not pd.isna(nose_y)) and (rw_y < nose_y):
        return "bowling"

    # fielding -> crouched: knee lower (higher y) than shoulder by threshold
    if not pd.isna(l_knee_y) and not pd.isna(l_sh_y) and (l_knee_y > l_sh_y + 0.05):
        return "fielding"
    if not pd.isna(r_knee_y) and not pd.isna(r_sh_y) and (r_knee_y > r_sh_y + 0.05):
        return "fielding"

    # batting -> standing (ankles roughly same y) and wrists near shoulder y
    if not pd.isna(l_ankle_y) and not pd.isna(r_ankle_y) and not pd.isna(l_sh_y) and not pd.isna(r_sh_y):
        ankle_diff = abs(l_ankle_y - r_ankle_y)
        sh_diff = abs(l_sh_y - r_sh_y)
        # wrists near shoulders?
        wrists_near_shoulders = (not pd.isna(lw_y) and not pd.isna(rw_y) and
                                  (abs(lw_y - l_sh_y) < 0.12 or abs(rw_y - r_sh_y) < 0.12))
        if ankle_diff < 0.06 and sh_diff < 0.06 and wrists_near_shoulders:
            return "batting"

    # otherwise assume follow-through
    return "follow_through"

# ========== segmentation ==========
def frames_to_segments(labels):
    """Group consecutive identical labels into segments (label,start,end inclusive)."""
    segments = []
    if not labels:
        return segments
    cur_label = labels[0][1]; start = labels[0][0]
    for frame_idx, lab in labels[1:]:
        if lab != cur_label:
            segments.append((cur_label, start, frame_idx - 1))
            cur_label = lab; start = frame_idx
    segments.append((cur_label, start, labels[-1][0]))
    # filter none and tiny
    out = []
    for lab, s, e in segments:
        if lab == "none":
            continue
        if (e - s + 1) >= MIN_SEGMENT_FRAMES:
            out.append((lab, s, e))
    return out

# ========== segment analysis ==========
def analyze_segment_metrics(df_segment):
    metrics = {}
    # ankle distance mean (stance width)
    ankle_dists = []
    for _, r in df_segment.iterrows():
        try:
            L = [r[f"x_{LEFT_ANKLE}"], r[f"y_{LEFT_ANKLE}"]]
            R = [r[f"x_{RIGHT_ANKLE}"], r[f"y_{RIGHT_ANKLE}"]]
            if not any(pd.isna(L)) and not any(pd.isna(R)):
                ankle_dists.append(euclidean(L, R))
        except Exception:
            continue
    metrics["ankle_distance_mean"] = float(np.mean(ankle_dists)) if ankle_dists else float("nan")

    # torso rotation proxy: mean abs difference between left and right torso angles
    torso_diffs = []
    for _, r in df_segment.iterrows():
        try:
            ls = [r[f"x_{LEFT_SHOULDER}"], r[f"y_{LEFT_SHOULDER}"]]
            rs = [r[f"x_{RIGHT_SHOULDER}"], r[f"y_{RIGHT_SHOULDER}"]]
            lh = [r[f"x_{LEFT_HIP}"], r[f"y_{LEFT_HIP}"]]
            rh = [r[f"x_{RIGHT_HIP}"], r[f"y_{RIGHT_HIP}"]]
            if not any(pd.isna(ls+rs+lh+rh)):
                angle_left = calculate_angle(ls, lh, rh)
                angle_right = calculate_angle(rs, rh, lh)
                torso_diffs.append(abs(angle_left - angle_right))
        except Exception:
            continue
    metrics["torso_rotation_proxy_mean"] = float(np.mean(torso_diffs)) if torso_diffs else float("nan")

    # elbow means
    left_elbow_angles = []
    right_elbow_angles = []
    for _, r in df_segment.iterrows():
        try:
            la = calculate_angle([r[f"x_{LEFT_SHOULDER}"], r[f"y_{LEFT_SHOULDER}"]],
                                 [r[f"x_{LEFT_ELBOW}"], r[f"y_{LEFT_ELBOW}"]],
                                 [r[f"x_{LEFT_WRIST}"], r[f"y_{LEFT_WRIST}"]])
            ra = calculate_angle([r[f"x_{RIGHT_SHOULDER}"], r[f"y_{RIGHT_SHOULDER}"]],
                                 [r[f"x_{RIGHT_ELBOW}"], r[f"y_{RIGHT_ELBOW}"]],
                                 [r[f"x_{RIGHT_WRIST}"], r[f"y_{RIGHT_WRIST}"]])
            if not math.isnan(la): left_elbow_angles.append(la)
            if not math.isnan(ra): right_elbow_angles.append(ra)
        except Exception:
            continue
    metrics["left_elbow_mean"] = float(np.mean(left_elbow_angles)) if left_elbow_angles else float("nan")
    metrics["right_elbow_mean"] = float(np.mean(right_elbow_angles)) if right_elbow_angles else float("nan")

    # right wrist frame-to-frame displacement mean (speed proxy)
    rw_disp = []
    prev = None
    for _, r in df_segment.iterrows():
        if pd.isna(r.get(f"x_{RIGHT_WRIST}", np.nan)):
            prev = None; continue
        cur = np.array([r[f"x_{RIGHT_WRIST}"], r[f"y_{RIGHT_WRIST}"]])
        if prev is not None:
            rw_disp.append(euclidean(cur, prev))
        prev = cur
    metrics["rw_mean_disp"] = float(np.mean(rw_disp)) if rw_disp else float("nan")

    return metrics

# ========== export helpers ==========
def export_segment_clip(original_video_path, start_frame, end_frame, out_path, fps):
    cap = cv2.VideoCapture(original_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for f in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    out.release()
    cap.release()

POSE_CONNECTIONS = [
    (11,13),(13,15),(12,14),(14,16),
    (11,12),(23,24),(11,23),(12,24),
    (23,25),(25,27),(24,26),(26,28)
]

def plot_skeleton_3d_row(row, ax):
    ax.clear()
    ax.set_xlim(0,1); ax.set_ylim(1,0); ax.set_zlim(-0.5,0.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    joints = [None]*POSE_LANDMARK_COUNT
    for i in range(POSE_LANDMARK_COUNT):
        try:
            x = float(row[f"x_{i}"]); y = float(row[f"y_{i}"]); z = float(row[f"z_{i}"])
            joints[i] = (x,y,z)
            ax.scatter(x,y,z,s=10,c='blue')
        except Exception:
            joints[i] = None
    for a,b in POSE_CONNECTIONS:
        if joints[a] and joints[b]:
            xa,ya,za = joints[a]; xb,yb,zb = joints[b]
            ax.plot([xa,xb],[ya,yb],[za,zb],c='black',linewidth=1.5)

def export_3d_skeleton_video_from_df(df_segment, out_video_path, fps=15):
    temp_dir = tempfile.mkdtemp()
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    image_files = []
    for idx, (_, row) in enumerate(df_segment.iterrows()):
        plot_skeleton_3d_row(row, ax)
        tmpfile = os.path.join(temp_dir, f"frame_{idx:04d}.png")
        plt.savefig(tmpfile, bbox_inches='tight', dpi=100)
        image_files.append(tmpfile)
    plt.close(fig)
    if not image_files:
        raise RuntimeError("No skeleton frames to save.")
    frame = cv2.imread(image_files[0])
    h,w,_ = frame.shape
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    for img in image_files:
        frame = cv2.imread(img)
        if frame is None:
            continue
        out.write(frame)
    out.release()
    shutil.rmtree(temp_dir)

# ========== main processing ==========
def process_video(video_path):
    print(f"\nProcessing: {video_path}")
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(OUTPUT_ROOT, base)
    os.makedirs(out_dir, exist_ok=True)

    df_all, fps = extract_all_keypoints(video_path)
    print(f"Extracted {len(df_all)} frames (fps={fps})")

    labels = [(int(idx), classify_frame(row)) for idx, row in df_all.iterrows()]
    segments = frames_to_segments(labels)
    print(f"Found {len(segments)} segments")

    for seg_idx, (label, s, e) in enumerate(segments):
        seg_name = f"segment_{seg_idx:03d}"
        seg_folder = os.path.join(out_dir, f"{label}_segments")
        os.makedirs(seg_folder, exist_ok=True)
        df_seg = df_all.iloc[s:e+1].reset_index(drop=True)

        # Save keypoints CSV
        csv_path = os.path.join(seg_folder, f"{seg_name}_keypoints.csv")
        df_seg.to_csv(csv_path, index=False)

        # Export clip from original video
        clip_path = os.path.join(seg_folder, f"{seg_name}_clip.mp4")
        export_segment_clip(video_path, s, e, clip_path, int(fps))

        # Export 3D skeleton animation
        skeleton_path = os.path.join(seg_folder, f"{seg_name}_skeleton.mp4")
        try:
            export_3d_skeleton_video_from_df(df_seg, skeleton_path, fps=min(15, int(fps)))
        except Exception as ex:
            print("Warning: failed to export skeleton video:", ex)

        # Analyze metrics and write summary
        metrics = analyze_segment_metrics(df_seg)
        summary_path = os.path.join(seg_folder, f"{seg_name}_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"video: {video_path}\nsegment: {seg_idx}\nlabel: {label}\nframes: {s}-{e}\n\n")
            f.write("Metrics:\n")
            for k,v in metrics.items():
                f.write(f"  {k}: {v}\n")
            f.write("\nComments:\n")
            if label == "batting":
                if not math.isnan(metrics.get("ankle_distance_mean", float("nan"))) and metrics["ankle_distance_mean"] < 0.05:
                    f.write("- Stance seems narrow; consider wider base for stability.\n")
                else:
                    f.write("- Stance width OK.\n")
                if not math.isnan(metrics.get("rw_mean_disp", float("nan"))) and metrics["rw_mean_disp"] > 0.02:
                    f.write("- Good wrist speed detected (likely strong bat swing).\n")
                else:
                    f.write("- Low wrist movement; check backlift/trigger.\n")
            elif label == "bowling":
                if not math.isnan(metrics.get("right_elbow_mean", float("nan"))) and metrics["right_elbow_mean"] > 160:
                    f.write("- Strong elbow extension at release (good pace mechanics).\n")
                else:
                    f.write("- Limited elbow extension; check release mechanics.\n")
                if not math.isnan(metrics.get("torso_rotation_proxy_mean", float("nan"))) and metrics["torso_rotation_proxy_mean"] > 5:
                    f.write("- Good torso rotation torque present.\n")
            elif label == "fielding":
                f.write("- Check dive posture and approach in the clip.\n")
            elif label == "follow_through":
                f.write("- Check balance and recovery; review skeleton clip.\n")

        print(f"Saved segment {seg_idx} [{label}] frames {s}-{e} -> {seg_folder}")

    print(f"Done. Outputs in {out_dir}")

if __name__ == "__main__":
    for vp in video_paths:
        if os.path.exists(vp):
            process_video(vp)
        else:
            print(f"Video not found: {vp}")
