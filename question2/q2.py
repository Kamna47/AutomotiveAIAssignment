#!/usr/bin/env python3
"""
DRIVER ACTIVITY MONITORING SYSTEM - JUPYTER NOTEBOOK VERSION
Complete standalone script for analyzing driver face videos

Run this entire cell in Jupyter Notebook to:
1. Process your face video
2. Generate CSV with frame-by-frame analysis
3. Create annotated video with visualizations
4. Save outputs to Downloads folder

Author: Enhanced from reference code
Date: February 2026
"""

# ============================================================================
# IMPORTS AND SETUP
# ============================================================================

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import math
import os
from pathlib import Path
from datetime import datetime
from IPython.display import display, HTML, Video
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DRIVER ACTIVITY MONITORING SYSTEM")
print("=" * 80)
print("\n📦 Loading libraries...")

# ============================================================================
# CONFIGURATION - CHANGE THESE VALUES
# ============================================================================

# YOUR VIDEO FILE PATH - UPDATE THIS!
VIDEO_PATH = r"C:\Users\akhilesh zende\Downloads\WIN_20260208_23_39_16_Pro.mp4"

# OUTPUT FOLDER - Will be created in Downloads
OUTPUT_FOLDER_NAME = "DriverActivityAnalysis_" + datetime.now().strftime("%Y%m%d_%H%M%S")

# Create output directory in Downloads folder
DOWNLOADS_PATH = str(Path.home() / "Downloads")
OUTPUT_DIR = os.path.join(DOWNLOADS_PATH, OUTPUT_FOLDER_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"✓ Output directory: {OUTPUT_DIR}")

# Output file paths
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "driver_activity_analysis.csv")
VIDEO_OUTPUT = os.path.join(OUTPUT_DIR, "annotated_video.mp4")
SUMMARY_OUTPUT = os.path.join(OUTPUT_DIR, "analysis_summary.txt")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clamp(x, minimum, maximum):
    """Clamp value between min and max"""
    return max(minimum, min(maximum, x))

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def safe_divide(num, denom, default=0.0):
    """Safe division avoiding divide by zero"""
    return num / denom if abs(denom) > 1e-6 else default

# ============================================================================
# EYE ANALYSIS FUNCTIONS
# ============================================================================

def calculate_eye_aspect_ratio(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    if len(eye_points) != 6:
        return 0.25
    
    p1, p2, p3, p4, p5, p6 = eye_points
    
    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    
    ear = safe_divide(vertical_1 + vertical_2, 2.0 * horizontal, 0.25)
    return ear

def classify_eye_state(ear, threshold=0.23):
    """Classify if eyes are open or closed"""
    return "eyeClosed" if ear < threshold else "eyeOpen"

# ============================================================================
# GAZE/PUPIL ANALYSIS FUNCTIONS
# ============================================================================

def calculate_iris_ratio(iris_points, left_corner, right_corner):
    """Calculate horizontal position of iris in eye (0=left, 0.5=center, 1=right)"""
    if len(iris_points) == 0:
        return 0.5
    
    iris_center = np.mean(iris_points, axis=0)
    eye_width = euclidean_distance(left_corner, right_corner)
    
    if eye_width < 1e-6:
        return 0.5
    
    iris_offset = iris_center[0] - left_corner[0]
    ratio = clamp(iris_offset / eye_width, 0.0, 1.0)
    
    return ratio

def classify_pupil_position(ratio):
    """Classify gaze direction based on iris position"""
    if ratio < 0.38:
        return "left"
    elif ratio > 0.62:
        return "right"
    else:
        return "center"

# ============================================================================
# HEAD POSE ESTIMATION
# ============================================================================

def estimate_head_pose(landmarks, img_width, img_height):
    """
    Estimate head pose angles (yaw, pitch, roll) using PnP algorithm
    Returns: (yaw, pitch, roll) in degrees
    """
    
    # Key facial points for pose estimation
    indices = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye": 33,
        "right_eye": 263,
        "left_mouth": 61,
        "right_mouth": 291
    }
    
    # 2D image points
    image_pts = np.array([
        [landmarks[indices["nose_tip"]].x * img_width, 
         landmarks[indices["nose_tip"]].y * img_height],
        [landmarks[indices["chin"]].x * img_width, 
         landmarks[indices["chin"]].y * img_height],
        [landmarks[indices["left_eye"]].x * img_width, 
         landmarks[indices["left_eye"]].y * img_height],
        [landmarks[indices["right_eye"]].x * img_width, 
         landmarks[indices["right_eye"]].y * img_height],
        [landmarks[indices["left_mouth"]].x * img_width, 
         landmarks[indices["left_mouth"]].y * img_height],
        [landmarks[indices["right_mouth"]].x * img_width, 
         landmarks[indices["right_mouth"]].y * img_height]
    ], dtype=np.float64)
    
    # 3D model points (generic face model)
    model_pts = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -63.6, -12.5),      # Chin
        (-43.3, 32.7, -26.0),     # Left eye
        (43.3, 32.7, -26.0),      # Right eye
        (-28.9, -28.9, -24.1),    # Left mouth
        (28.9, -28.9, -24.1)      # Right mouth
    ], dtype=np.float64)
    
    # Camera matrix
    focal_length = img_width
    center = (img_width / 2, img_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros((4, 1))
    
    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(
        model_pts, image_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return 0.0, 0.0, 0.0
    
    # Convert to rotation matrix
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    
    # Extract Euler angles
    sy = math.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
        y = math.atan2(-rot_mat[2, 0], sy)
        z = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        x = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
        y = math.atan2(-rot_mat[2, 0], sy)
        z = 0
    
    pitch = np.degrees(x)
    yaw = np.degrees(y)
    roll = np.degrees(z)
    
    return yaw, pitch, roll

# ============================================================================
# CLASSIFICATION FUNCTIONS
# ============================================================================

def classify_head_rotation(yaw):
    """Classify horizontal head rotation"""
    if yaw > 35:
        return "overLeftShoulder"
    elif yaw > 15:
        return "45DegreeLeft"
    elif yaw < -35:
        return "overRightShoulder"
    elif yaw < -15:
        return "45DegreeRight"
    else:
        return "center"

def classify_head_tilt_front(pitch):
    """Classify forward/backward head tilt"""
    if pitch < -20:
        return "extremeBackTilt"
    elif pitch < -10:
        return "backTilt"
    elif pitch > 20:
        return "extremeFrontTilt"
    elif pitch > 10:
        return "frontTilt"
    else:
        return "straightFrontAhead"

def classify_head_tilt_side(roll):
    """Classify left/right head tilt"""
    if roll > 15:
        return "headTopRight_NeckLeft"
    elif roll < -15:
        return "headTopLeft_NeckRight"
    else:
        return "straight"

def classify_driving_status(yaw, pitch, blink_rate, gaze_ratio, avg_ear):
    """Classify overall driver attention status"""
    
    # Distracted: looking away significantly
    if abs(yaw) > 25 or abs(pitch) > 25:
        return "distracted"
    
    # Drowsy: eyes mostly closed or very high blink
    if avg_ear < 0.20 or blink_rate > 35:
        return "drowsy"
    
    # Extreme focus: very stable and centered
    if abs(yaw) < 8 and abs(pitch) < 8 and gaze_ratio > 0.75 and blink_rate < 12:
        return "extremeFocus"
    
    # Focus: good attention
    if gaze_ratio > 0.60 and abs(yaw) < 15:
        return "Focus"
    
    # Relaxed/bored: high blink but not distracted
    if blink_rate > 25:
        return "Relaxed/bore"
    
    return "Normal"

# ============================================================================
# BLINK DETECTOR CLASS
# ============================================================================

class BlinkDetector:
    """Tracks blink events and calculates blink rate"""
    
    def __init__(self, window_sec=60.0):
        self.blinks = []
        self.was_closed = False
        self.window = window_sec
    
    def update(self, eye_state, timestamp):
        """Update with current eye state"""
        if eye_state == "eyeClosed" and not self.was_closed:
            self.was_closed = True
        elif eye_state == "eyeOpen" and self.was_closed:
            self.was_closed = False
            self.blinks.append(timestamp)
        
        # Remove old events
        self.blinks = [t for t in self.blinks if timestamp - t <= self.window]
    
    def get_rate(self):
        """Get blinks per minute"""
        return float(len(self.blinks))

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_driver_video(video_path, csv_path, video_out_path):
    """
    Main processing function
    """
    
    print("\n" + "=" * 80)
    print("STARTING VIDEO ANALYSIS")
    print("=" * 80)
    
    # Check video exists
    if not os.path.exists(video_path):
        print(f"\n❌ ERROR: Video not found at: {video_path}")
        return None
    
    print(f"\n📹 Input: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ ERROR: Cannot open video file")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30.0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Resolution: {width}x{height}")
    print(f"📊 FPS: {fps:.2f}")
    print(f"📊 Total frames: {total_frames}")
    print(f"📊 Duration: {total_frames/fps:.2f} seconds")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
    
    # Initialize MediaPipe
    print("\n🔧 Initializing MediaPipe FaceMesh...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Landmark indices
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]
    
    # Initialize tracking
    blink_detector = BlinkDetector()
    gaze_center_count = 0
    frames_with_face = 0
    results_list = []
    
    print("\n🚀 Processing frames...")
    start_time = time.time()
    
    frame_num = 0
    last_percent = -1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        timestamp = frame_num / fps
        
        # Progress indicator
        percent = int((frame_num / total_frames) * 100)
        if percent != last_percent and percent % 5 == 0:
            print(f"   ⏳ {percent}% complete ({frame_num}/{total_frames} frames)")
            last_percent = percent
        
        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        
        # Default values
        eye_state = "noFace"
        pupil_pos = "noFace"
        drive_status = "noFace"
        head_rot = "noFace"
        head_front = "noFace"
        head_side = "noFace"
        yaw = pitch = roll = 0.0
        ear = 0.0
        blink_rate = 0.0
        
        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            frames_with_face += 1
            
            # === EYE STATE ===
            left_eye_pts = [(lm[i].x * width, lm[i].y * height) for i in LEFT_EYE]
            right_eye_pts = [(lm[i].x * width, lm[i].y * height) for i in RIGHT_EYE]
            
            ear_l = calculate_eye_aspect_ratio(left_eye_pts)
            ear_r = calculate_eye_aspect_ratio(right_eye_pts)
            ear = (ear_l + ear_r) / 2.0
            
            eye_state = classify_eye_state(ear)
            blink_detector.update(eye_state, timestamp)
            blink_rate = blink_detector.get_rate()
            
            # === PUPIL POSITION ===
            left_iris_pts = np.array([(lm[i].x * width, lm[i].y * height) for i in LEFT_IRIS])
            left_corner = np.array([lm[33].x * width, lm[33].y * height])
            right_corner = np.array([lm[133].x * width, lm[133].y * height])
            
            iris_ratio = calculate_iris_ratio(left_iris_pts, left_corner, right_corner)
            pupil_pos = classify_pupil_position(iris_ratio)
            
            if pupil_pos == "center":
                gaze_center_count += 1
            
            gaze_ratio = gaze_center_count / max(frames_with_face, 1)
            
            # === HEAD POSE ===
            yaw, pitch, roll = estimate_head_pose(lm, width, height)
            head_rot = classify_head_rotation(yaw)
            head_front = classify_head_tilt_front(pitch)
            head_side = classify_head_tilt_side(roll)
            
            # === DRIVING STATUS ===
            drive_status = classify_driving_status(yaw, pitch, blink_rate, gaze_ratio, ear)
            
            # === DRAW ANNOTATIONS ===
            y_pos = 30
            line_h = 30
            
            # Eye state
            color = (0, 255, 0) if eye_state == "eyeOpen" else (0, 165, 255)
            cv2.putText(frame, f"a1 Eye: {eye_state}", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            y_pos += line_h
            
            # Pupil
            cv2.putText(frame, f"a2 Pupil: {pupil_pos}", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            y_pos += line_h
            
            # Status
            status_colors = {
                "extremeFocus": (0, 255, 0),
                "Focus": (0, 200, 0),
                "Normal": (0, 255, 255),
                "Relaxed/bore": (0, 165, 255),
                "distracted": (0, 100, 255),
                "drowsy": (0, 0, 255)
            }
            color = status_colors.get(drive_status, (255, 255, 255))
            cv2.putText(frame, f"a3 Status: {drive_status}", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            y_pos += line_h
            
            # Head rotation
            cv2.putText(frame, f"a4 Rotation: {head_rot}", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2)
            y_pos += line_h
            
            # Front tilt
            cv2.putText(frame, f"a5 Front: {head_front}", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2)
            y_pos += line_h
            
            # Side tilt
            cv2.putText(frame, f"a6 Side: {head_side}", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2)
            y_pos += line_h
            
            # Angles
            cv2.putText(frame, f"Yaw:{yaw:.1f} Pitch:{pitch:.1f} Roll:{roll:.1f}", 
                       (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            y_pos += line_h
            
            # Stats
            cv2.putText(frame, f"Blink:{blink_rate:.1f}/min EAR:{ear:.3f}", 
                       (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Save results
        results_list.append({
            "frame": frame_num,
            "time_sec": round(timestamp, 3),
            "a1_eye_state": eye_state,
            "a2_pupil_position": pupil_pos,
            "a3_driving_status": drive_status,
            "a4_head_rotation": head_rot,
            "a5_head_front_tilt": head_front,
            "a6_head_side_tilt": head_side,
            "yaw_deg": round(yaw, 2),
            "pitch_deg": round(pitch, 2),
            "roll_deg": round(roll, 2),
            "ear": round(ear, 3),
            "blink_rate_per_min": round(blink_rate, 1)
        })
        
        # Write frame
        video_writer.write(frame)
    
    # Cleanup
    cap.release()
    video_writer.release()
    face_mesh.close()
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Processing complete!")
    print(f"   ⏱️  Time: {elapsed:.2f} seconds")
    print(f"   📊 Frames processed: {frame_num}")
    print(f"   📊 Frames with face: {frames_with_face}")
    
    # Save CSV
    df = pd.DataFrame(results_list)
    df.to_csv(csv_path, index=False)
    print(f"\n💾 CSV saved: {csv_path}")
    print(f"💾 Video saved: {video_out_path}")
    
    return df

# ============================================================================
# ANALYSIS AND SUMMARY FUNCTIONS
# ============================================================================

def generate_summary(df, output_path):
    """Generate text summary of analysis"""
    
    summary = []
    summary.append("=" * 80)
    summary.append("DRIVER ACTIVITY ANALYSIS SUMMARY")
    summary.append("=" * 80)
    summary.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"Total frames: {len(df)}")
    
    # Filter only frames with face detected
    df_face = df[df['a1_eye_state'] != 'noFace']
    summary.append(f"Frames with face: {len(df_face)} ({len(df_face)/len(df)*100:.1f}%)")
    
    if len(df_face) > 0:
        summary.append("\n" + "-" * 80)
        summary.append("METRIC DISTRIBUTIONS")
        summary.append("-" * 80)
        
        # a1: Eye State
        summary.append("\na1. EYE STATE:")
        eye_counts = df_face['a1_eye_state'].value_counts()
        for state, count in eye_counts.items():
            pct = count / len(df_face) * 100
            summary.append(f"   {state}: {count} frames ({pct:.1f}%)")
        
        # a2: Pupil Position
        summary.append("\na2. PUPIL POSITION:")
        pupil_counts = df_face['a2_pupil_position'].value_counts()
        for pos, count in pupil_counts.items():
            pct = count / len(df_face) * 100
            summary.append(f"   {pos}: {count} frames ({pct:.1f}%)")
        
        # a3: Driving Status
        summary.append("\na3. DRIVING STATUS:")
        status_counts = df_face['a3_driving_status'].value_counts()
        for status, count in status_counts.items():
            pct = count / len(df_face) * 100
            summary.append(f"   {status}: {count} frames ({pct:.1f}%)")
        
        # a4: Head Rotation
        summary.append("\na4. HEAD ROTATION:")
        rot_counts = df_face['a4_head_rotation'].value_counts()
        for rot, count in rot_counts.items():
            pct = count / len(df_face) * 100
            summary.append(f"   {rot}: {count} frames ({pct:.1f}%)")
        
        # a5: Head Front Tilt
        summary.append("\na5. HEAD FRONT TILT:")
        front_counts = df_face['a5_head_front_tilt'].value_counts()
        for tilt, count in front_counts.items():
            pct = count / len(df_face) * 100
            summary.append(f"   {tilt}: {count} frames ({pct:.1f}%)")
        
        # a6: Head Side Tilt
        summary.append("\na6. HEAD SIDE TILT:")
        side_counts = df_face['a6_head_side_tilt'].value_counts()
        for tilt, count in side_counts.items():
            pct = count / len(df_face) * 100
            summary.append(f"   {tilt}: {count} frames ({pct:.1f}%)")
        
        # Statistics
        summary.append("\n" + "-" * 80)
        summary.append("STATISTICS")
        summary.append("-" * 80)
        
        summary.append(f"\nAverage Eye Aspect Ratio: {df_face['ear'].mean():.3f}")
        summary.append(f"Average Blink Rate: {df_face['blink_rate_per_min'].mean():.1f} per minute")
        summary.append(f"\nHead Pose Angles (average):")
        summary.append(f"   Yaw: {df_face['yaw_deg'].mean():.2f}°")
        summary.append(f"   Pitch: {df_face['pitch_deg'].mean():.2f}°")
        summary.append(f"   Roll: {df_face['roll_deg'].mean():.2f}°")
    
    summary.append("\n" + "=" * 80)
    
    summary_text = "\n".join(summary)
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(summary_text)
    
    return summary_text

def display_sample_data(df, n=10):
    """Display sample rows from dataframe"""
    print("\n" + "=" * 80)
    print(f"SAMPLE DATA (first {n} rows with face detected)")
    print("=" * 80)
    
    df_face = df[df['a1_eye_state'] != 'noFace'].head(n)
    
    if len(df_face) > 0:
        display(df_face)
    else:
        print("No frames with face detected in first rows")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"\n📹 Video input: {VIDEO_PATH}")
    print(f"📁 Output folder: {OUTPUT_DIR}")
    print(f"📄 CSV output: {CSV_OUTPUT}")
    print(f"🎥 Video output: {VIDEO_OUTPUT}")
    print(f"📝 Summary output: {SUMMARY_OUTPUT}")
    
    # Process video
    df = process_driver_video(VIDEO_PATH, CSV_OUTPUT, VIDEO_OUTPUT)
    
    if df is None:
        print("\n❌ Processing failed!")
        return
    
    # Generate summary
    print("\n📝 Generating summary...")
    summary = generate_summary(df, SUMMARY_OUTPUT)
    print(summary)
    
    # Display sample data
    display_sample_data(df, n=10)
    
    # Final message
    print("\n" + "=" * 80)
    print("✅ ALL DONE!")
    print("=" * 80)
    print(f"\n📁 All outputs saved to: {OUTPUT_DIR}")
    print(f"\n   1. {os.path.basename(CSV_OUTPUT)} - Frame-by-frame data")
    print(f"   2. {os.path.basename(VIDEO_OUTPUT)} - Annotated video")
    print(f"   3. {os.path.basename(SUMMARY_OUTPUT)} - Analysis summary")
    print("\n" + "=" * 80)
    
    # Try to display video in notebook
    try:
        print("\n🎥 Video preview:")
        display(Video(VIDEO_OUTPUT, width=640))
    except:
        print("\n(Video display not available in this environment)")
    
    return df

# ============================================================================
# RUN THE ANALYSIS
# ============================================================================

if __name__ == "__main__":
    # Run the complete analysis
    results_df = main()

# If running in Jupyter, just execute:
# results_df = main()
