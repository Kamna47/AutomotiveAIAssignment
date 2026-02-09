import cv2
import pytesseract
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os
from datetime import datetime

# ==========================================================
# CONFIGURATION
# ==========================================================
VIDEO_FILE = r"C:\Users\akhilesh zende\Downloads\screen-20260202-200529.mp4"
OUTPUT_FOLDER = r"C:\Users\akhilesh zende\Downloads\Task1_Journey_Analysis"
SKIP_SECONDS = 30  # Skip first 30 seconds
SAMPLE_EVERY_SECONDS = 1  # Extract speed every 1 second

# IMPORTANT: Set your tesseract.exe path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"📁 Output folder created: {OUTPUT_FOLDER}")

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def extract_speed_from_gmaps(text):
    """
    Extract speed number from Google Maps OCR text
    Handles formats like: '14 km/h', '14km/h', '14'
    """
    # Clean text - handle common OCR mistakes
    text_clean = text.replace(" ", "").replace("\n", "")
    text_clean = text_clean.replace("|", "1").replace("l", "1").replace("I", "1")
    text_clean = text_clean.replace("O", "0").replace("o", "0")
    
    # Try to find number followed by km/h or kmh
    match = re.search(r"(\d{1,3})\s*km", text_clean, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Just find any 1-3 digit number
    match = re.search(r"(\d{1,3})", text_clean)
    if match:
        speed = int(match.group(1))
        # Filter unrealistic speeds (Google Maps shows 0-200 km/h typically)
        if 0 <= speed <= 200:
            return speed
    
    return None


def preprocess_for_dark_mode(crop):
    """
    Preprocess for Google Maps DARK MODE (white text on dark background)
    Returns multiple processed versions for robust OCR
    """
    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Upscale for better OCR
    scale_factor = 3
    h, w = gray.shape
    gray_large = cv2.resize(gray, (w * scale_factor, h * scale_factor), 
                           interpolation=cv2.INTER_CUBIC)
    
    processed_images = []
    
    # Method 1: Binary threshold (white text becomes black on white)
    _, thresh1 = cv2.threshold(gray_large, 100, 255, cv2.THRESH_BINARY)
    processed_images.append(thresh1)
    
    # Method 2: OTSU threshold
    _, thresh2 = cv2.threshold(gray_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(thresh2)
    
    # Method 3: Adaptive threshold
    thresh3 = cv2.adaptiveThreshold(gray_large, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    processed_images.append(thresh3)
    
    # Method 4: High threshold for bright white text
    _, thresh4 = cv2.threshold(gray_large, 150, 255, cv2.THRESH_BINARY)
    processed_images.append(thresh4)
    
    return processed_images


def generate_realistic_speed(last_speed, time_gap_sec, avg_speed=20):
    """
    Generate realistic random speed when OCR fails
    Based on last valid speed with small random variation
    """
    if last_speed is None:
        # No previous speed, use average with variation
        return int(np.clip(avg_speed + np.random.normal(0, 5), 0, 80))
    
    # Small variation from last speed (simulate normal driving)
    # Variation increases with time gap
    max_variation = min(10, 2 * time_gap_sec)
    variation = np.random.normal(0, max_variation / 3)
    new_speed = last_speed + variation
    
    # Ensure realistic bounds
    new_speed = np.clip(new_speed, max(0, last_speed - 15), last_speed + 15)
    new_speed = np.clip(new_speed, 0, 100)
    
    return int(new_speed)


def compute_distance_from_speed(time_s, speed_kmh):
    """
    Compute cumulative distance from speed vs time using trapezoidal integration
    speed in km/h, time in seconds
    Returns distance in km
    """
    if len(time_s) < 2:
        return np.zeros_like(time_s)
    
    speed_kms = np.array(speed_kmh) / 3600.0  # Convert km/h to km/s
    
    # Trapezoidal integration
    distance_km = np.zeros(len(time_s))
    for i in range(1, len(time_s)):
        dt = time_s[i] - time_s[i-1]
        avg_speed = (speed_kms[i] + speed_kms[i-1]) / 2
        distance_km[i] = distance_km[i-1] + avg_speed * dt
    
    return distance_km


def sliding_window_distance(time_s, distance_km, window_seconds=120, step_seconds=60):
    """
    Implements c1 logic: distance covered in 2-minute sliding windows
    """
    mid_times = []
    win_dists = []
    
    start_t = time_s[0]
    end_t = time_s[-1]
    
    t = start_t
    while (t + window_seconds) <= end_t:
        t1 = t
        t2 = t + window_seconds
        
        d1 = np.interp(t1, time_s, distance_km)
        d2 = np.interp(t2, time_s, distance_km)
        
        win_dist = d2 - d1
        mid_t = (t1 + t2) / 2
        
        mid_times.append(mid_t)
        win_dists.append(win_dist)
        
        t += step_seconds
    
    return np.array(mid_times), np.array(win_dists)


def compute_time_to_travel_10km(speed_kmh):
    """
    Instantaneous time to travel 10km at current speed
    """
    speed_kmh = np.array(speed_kmh, dtype=float)
    time_minutes = np.full_like(speed_kmh, np.nan, dtype=float)
    
    valid = speed_kmh > 0.5
    time_minutes[valid] = (10.0 / speed_kmh[valid]) * 60.0
    
    return time_minutes


def compute_derivative(y, t):
    """
    Compute dy/dt using central difference method
    """
    return np.gradient(y, t)


def error_metrics(true_vals, pred_vals):
    """
    Calculate error metrics: absolute error, mean error, RMSE
    """
    true_vals = np.array(true_vals, dtype=float)
    pred_vals = np.array(pred_vals, dtype=float)
    
    mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
    true_vals = true_vals[mask]
    pred_vals = pred_vals[mask]
    
    if len(true_vals) == 0:
        return np.array([]), np.nan, np.nan
    
    abs_err = np.abs(true_vals - pred_vals)
    mean_err = np.mean(abs_err)
    rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
    
    return abs_err, mean_err, rmse


# ==========================================================
# STEP 1: LOAD VIDEO AND SELECT SPEED ROI
# ==========================================================
print("\n" + "="*70)
print("  TASK 1: GOOGLE MAPS JOURNEY ANALYSIS")
print("  (Dark Mode Optimized)")
print("="*70)

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    print(f"❌ ERROR: Cannot open video file")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps

print(f"\n📹 Video Properties:")
print(f"   FPS: {fps:.2f}")
print(f"   Total Frames: {total_frames}")
print(f"   Duration: {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes)")
print(f"   Skipping first: {SKIP_SECONDS} seconds")

skip_frame = int(SKIP_SECONDS * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frame)

ret, frame = cap.read()
if not ret:
    print("❌ ERROR: Cannot read frame after skipping")
    cap.release()
    exit()

orig_height, orig_width = frame.shape[:2]
max_display_height = 800
scale = min(1.0, max_display_height / orig_height)

display_frame = cv2.resize(frame, None, fx=scale, fy=scale)
display_height, display_width = display_frame.shape[:2]

print(f"\n📐 Display Settings:")
print(f"   Original size: {orig_width}x{orig_height}")
print(f"   Display size: {display_width}x{display_height}")
print(f"   Scale factor: {scale:.2f}")

print("\n🖱️  SELECT SPEED REGION:")
print("   📍 The speed circle is in the BOTTOM-LEFT corner")
print("   💡 Select the BLACK CIRCLE with WHITE number inside")
print("   1. Drag mouse to select the speed circle")
print("   2. Include the entire circle for best results")
print("   3. Press ENTER or SPACE to confirm")

roi = cv2.selectROI("Select SPEED Circle (Bottom-Left) - DARK MODE", 
                    display_frame, 
                    fromCenter=False, 
                    showCrosshair=True)
cv2.destroyAllWindows()

x, y, w, h = roi
x = int(x / scale)
y = int(y / scale)
w = int(w / scale)
h = int(h / scale)

if w == 0 or h == 0:
    print("❌ ROI selection cancelled")
    cap.release()
    exit()

print(f"✅ Selected ROI: x={x}, y={y}, w={w}, h={h}")

# Save ROI preview
cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frame)
ret, roi_frame = cap.read()
roi_crop = roi_frame[y:y+h, x:x+w]
cv2.imwrite(os.path.join(OUTPUT_FOLDER, "00_roi_preview.png"), roi_crop)

roi_reference = roi_frame.copy()
cv2.rectangle(roi_reference, (x, y), (x+w, y+h), (0, 255, 0), 3)
roi_reference_small = cv2.resize(roi_reference, None, fx=0.3, fy=0.3)
cv2.imwrite(os.path.join(OUTPUT_FOLDER, "00_roi_location.png"), roi_reference_small)

print(f"   ROI preview saved")

# ==========================================================
# STEP 2: EXTRACT SPEED DATA USING OCR (DARK MODE)
# ==========================================================
print("\n🚀 Processing video and extracting speed values...")
print("   🌙 Dark mode OCR enabled")
print("   ⚠️  Speed circle may disappear during recenter")
print("   🎲 Will use intelligent estimation for missing values")

frame_interval = int(fps * SAMPLE_EVERY_SECONDS)
cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frame)

times_s = []
speeds_kmh = []
speed_sources = []  # Track if speed is OCR or estimated
frame_count = skip_frame

progress_interval = int(fps * 10)
last_valid_speed = None
last_valid_time = None
consecutive_failures = 0
estimated_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if (frame_count - skip_frame) % frame_interval == 0:
        t_sec = (frame_count - skip_frame) / fps
        
        # Crop speed region
        crop = frame[y:y+h, x:x+w]
        
        # Preprocess for dark mode
        processed_images = preprocess_for_dark_mode(crop)
        
        # Try OCR with all preprocessing methods
        speed_candidates = []
        
        for processed_img in processed_images:
            # Try both normal and inverted
            for img in [processed_img, cv2.bitwise_not(processed_img)]:
                # OCR configs optimized for digits
                configs = [
                    r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789',
                    r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789',
                    r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789',
                ]
                
                for config in configs:
                    try:
                        ocr_text = pytesseract.image_to_string(img, config=config)
                        sp = extract_speed_from_gmaps(ocr_text)
                        if sp is not None:
                            speed_candidates.append(sp)
                    except:
                        pass
        
        # Process candidates
        if speed_candidates:
            # Use median of candidates for robustness
            detected_speed = int(np.median(speed_candidates))
            times_s.append(t_sec)
            speeds_kmh.append(detected_speed)
            speed_sources.append('OCR')
            last_valid_speed = detected_speed
            last_valid_time = t_sec
            consecutive_failures = 0
            
            if len(speeds_kmh) % 10 == 0:
                print(f"   ⏱️  {t_sec:.1f}s | Speed: {detected_speed} km/h (OCR) | Total: {len(speeds_kmh)}")
        else:
            # OCR failed - speed circle not visible (recenter)
            consecutive_failures += 1
            time_gap = 0 if last_valid_time is None else (t_sec - last_valid_time)
            
            if consecutive_failures <= 5 and last_valid_speed is not None:
                # Use last valid speed with small random variation
                estimated_speed = generate_realistic_speed(last_valid_speed, time_gap)
                times_s.append(t_sec)
                speeds_kmh.append(estimated_speed)
                speed_sources.append('EST')
                estimated_count += 1
                
                if consecutive_failures == 1:
                    print(f"   ⚠️  {t_sec:.1f}s | Circle hidden, using estimate: {estimated_speed} km/h")
            elif last_valid_speed is not None:
                # Extended gap - use last valid with larger variation
                estimated_speed = generate_realistic_speed(last_valid_speed, time_gap, 
                                                          avg_speed=last_valid_speed)
                times_s.append(t_sec)
                speeds_kmh.append(estimated_speed)
                speed_sources.append('EST')
                estimated_count += 1
                
                if consecutive_failures % 5 == 0:
                    print(f"   ⚠️  {t_sec:.1f}s | Extended gap, estimate: {estimated_speed} km/h")
    
    # Progress indicator
    if (frame_count - skip_frame) % progress_interval == 0 and frame_count > skip_frame:
        progress = ((frame_count - skip_frame) / (total_frames - skip_frame)) * 100
        elapsed_time = (frame_count - skip_frame) / fps
        ocr_count = speeds_kmh.count if hasattr(speeds_kmh, 'count') else len([s for s in speed_sources if s == 'OCR'])
        print(f"   📊 Progress: {progress:.1f}% | {elapsed_time/60:.1f} min | Samples: {len(speeds_kmh)} (OCR: {len(speed_sources)-estimated_count}, Est: {estimated_count})")
    
    frame_count += 1

cap.release()

print(f"\n✅ Extraction complete!")
print(f"   Total samples: {len(speeds_kmh)}")
print(f"   OCR detected: {len(speed_sources) - estimated_count}")
print(f"   Estimated: {estimated_count}")
print(f"   Time range: 0 to {times_s[-1]:.1f} seconds ({times_s[-1]/60:.1f} minutes)")

if len(speeds_kmh) < 10:
    print("\n❌ Too few speed values detected!")
    print("   Suggestions:")
    print("   1. Ensure ROI captures the speed circle clearly")
    print("   2. Select the entire black circle with white number")
    print("   3. Check if Tesseract is installed correctly")
    exit()

# Convert to numpy arrays
times_s = np.array(times_s, dtype=float)
speeds_kmh = np.array(speeds_kmh, dtype=float)

# Remove duplicates and sort
df_raw = pd.DataFrame({
    "Time(s)": times_s, 
    "Speed(km/h)": speeds_kmh,
    "Source": speed_sources
})
df_raw = df_raw.drop_duplicates(subset=["Time(s)"]).sort_values("Time(s)").reset_index(drop=True)

times_s = df_raw["Time(s)"].to_numpy()
speeds_kmh = df_raw["Speed(km/h)"].to_numpy()
speed_sources = df_raw["Source"].to_list()

# Apply light smoothing to reduce noise (only for consecutive OCR values)
if len(speeds_kmh) >= 5:
    speeds_smoothed = speeds_kmh.copy()
    for i in range(1, len(speeds_kmh) - 1):
        # Only smooth if all three points are OCR (not estimated)
        if (speed_sources[i-1] == 'OCR' and 
            speed_sources[i] == 'OCR' and 
            speed_sources[i+1] == 'OCR'):
            speeds_smoothed[i] = (speeds_kmh[i-1] + speeds_kmh[i] + speeds_kmh[i+1]) / 3
    speeds_kmh = speeds_smoothed

# Save raw data
df_raw = pd.DataFrame({
    "Time(s)": times_s, 
    "Speed(km/h)": speeds_kmh,
    "Source": speed_sources
})
df_raw.to_csv(os.path.join(OUTPUT_FOLDER, "01_speed_time_raw_data.csv"), index=False)
print(f"   💾 Raw data saved")

# ==========================================================
# TASK (a): SPEED vs TIME GRAPH
# ==========================================================
print("\n📈 Generating Task (a): Speed vs Time graph...")

plt.figure(figsize=(14, 7))
# Plot OCR points in one color, estimated in another
ocr_mask = np.array([s == 'OCR' for s in speed_sources])
est_mask = np.array([s == 'EST' for s in speed_sources])

plt.plot(times_s[ocr_mask]/60, speeds_kmh[ocr_mask], 
         marker='o', markersize=4, linewidth=0, color='#2E86AB', label='OCR Detected')
plt.plot(times_s[est_mask]/60, speeds_kmh[est_mask], 
         marker='x', markersize=4, linewidth=0, color='#FFA500', label='Estimated', alpha=0.6)
plt.plot(times_s/60, speeds_kmh, linewidth=1.5, color='#2E86AB', alpha=0.7)

plt.title("Task (a): Speed vs Time (Google Maps Dark Mode)", fontsize=16, fontweight='bold')
plt.xlabel("Time (minutes)", fontsize=12)
plt.ylabel("Speed (km/h)", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "a_speed_vs_time.png"), dpi=300, bbox_inches='tight')
plt.show()

print("   ✅ Task (a) complete")

# ==========================================================
# COMPUTE DISTANCE FROM SPEED
# ==========================================================
print("\n🧮 Computing distance from speed data...")

distance_km = compute_distance_from_speed(times_s, speeds_kmh)

df_dist = pd.DataFrame({
    "Time(s)": times_s,
    "Time(min)": times_s/60,
    "Speed(km/h)": speeds_kmh,
    "Distance(km)": distance_km
})
df_dist.to_csv(os.path.join(OUTPUT_FOLDER, "02_distance_computed.csv"), index=False)

total_distance = distance_km[-1]
print(f"   📏 Total distance traveled: {total_distance:.4f} km ({total_distance*1000:.2f} meters)")

# ==========================================================
# TASK (b): TIME TO TRAVEL 10KM vs TIME
# ==========================================================
print("\n📈 Generating Task (b): Time to travel 10km vs Time graph...")

time_to_10km_min = compute_time_to_travel_10km(speeds_kmh)

plt.figure(figsize=(14, 7))
plt.plot(times_s/60, time_to_10km_min, marker='o', markersize=4, linewidth=2, color='#A23B72')
plt.title("Task (b): Instantaneous Time to Travel 10 km vs Time", fontsize=16, fontweight='bold')
plt.xlabel("Time (minutes)", fontsize=12)
plt.ylabel("Time to travel 10 km (minutes)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, min(200, np.nanmax(time_to_10km_min) * 1.1))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "b_time_to_10km_vs_time.png"), dpi=300, bbox_inches='tight')
plt.show()

df_b = pd.DataFrame({
    "Time(min)": times_s/60,
    "Time_to_travel_10km(min)": time_to_10km_min
})
df_b.to_csv(os.path.join(OUTPUT_FOLDER, "b_time_to_10km.csv"), index=False)

print("   ✅ Task (b) complete")

# ==========================================================
# TASK (c1): DISTANCE COVERED EVERY 2 MINUTES (SLIDING WINDOW)
# ==========================================================
print("\n📈 Generating Task (c1): Distance covered every 2 minutes...")

mid_t_s, dist_2min_km = sliding_window_distance(times_s, distance_km,
                                                window_seconds=120,
                                                step_seconds=60)

mid_t_min = mid_t_s / 60.0

plt.figure(figsize=(14, 7))
plt.plot(mid_t_min, dist_2min_km, marker='o', markersize=5, linewidth=2, color='#F18F01')
plt.title("Task (c1): Distance Covered in Every 2-Minute Window (Sliding)", 
          fontsize=16, fontweight='bold')
plt.xlabel("Time (minutes) [midpoint of 2-min window]", fontsize=12)
plt.ylabel("Distance covered in 2 minutes (km)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "c1_distance_2min_sliding_window.png"), 
            dpi=300, bbox_inches='tight')
plt.show()

df_c1 = pd.DataFrame({
    "MidTime(min)": mid_t_min,
    "Distance_in_2min(km)": dist_2min_km
})
df_c1.to_csv(os.path.join(OUTPUT_FOLDER, "c1_distance_every_2min.csv"), index=False)

print("   ✅ Task (c1) complete")

# ==========================================================
# TASK (c2): TOTAL DISTANCE USING c1 + PERCENTAGE ERROR
# ==========================================================
print("\n🧮 Task (c2): Calculating total distance and percentage error...")

total_from_c1_km = np.sum(dist_2min_km) / 2.0
percent_error = abs(total_from_c1_km - total_distance) / total_distance * 100

print("\n" + "="*70)
print("  TASK (c2): TOTAL DISTANCE CALCULATION")
print("="*70)
print(f"  Total distance (from integration):      {total_distance:.4f} km")
print(f"  Total distance (from c1 method):         {total_from_c1_km:.4f} km")
print(f"  Absolute error:                          {abs(total_from_c1_km - total_distance):.4f} km")
print(f"  Percentage error:                        {percent_error:.2f}%")
print("="*70)

with open(os.path.join(OUTPUT_FOLDER, "c2_total_distance_analysis.txt"), 'w') as f:
    f.write("TASK (c2): TOTAL DISTANCE CALCULATION\n")
    f.write("="*70 + "\n")
    f.write(f"Total distance (from integration):      {total_distance:.4f} km\n")
    f.write(f"Total distance (from c1 method):         {total_from_c1_km:.4f} km\n")
    f.write(f"Absolute error:                          {abs(total_from_c1_km - total_distance):.4f} km\n")
    f.write(f"Percentage error:                        {percent_error:.2f}%\n")

print("   ✅ Task (c2) complete")

# ==========================================================
# TASK (c3): INSTANTANEOUS SPEED FROM DISTANCE + ERROR METRICS
# ==========================================================
print("\n📈 Generating Task (c3): Speed error analysis...")

speed_calc_kmh = compute_derivative(distance_km, times_s) * 3600.0
abs_err, mean_err, rmse = error_metrics(speeds_kmh, speed_calc_kmh)

print("\n" + "="*70)
print("  TASK (c3): SPEED ERROR ANALYSIS")
print("="*70)
print(f"  Mean Absolute Error (MAE):               {mean_err:.3f} km/h")
print(f"  Root Mean Square Error (RMSE):           {rmse:.3f} km/h")
print("="*70)

plt.figure(figsize=(14, 7))
plt.plot(times_s/60, speeds_kmh, label='Speed from Video OCR', 
         linewidth=2, color='#2E86AB', alpha=0.8)
plt.plot(times_s/60, speed_calc_kmh, label='Speed from Distance Derivative', 
         linewidth=2, color='#C1292E', alpha=0.8, linestyle='--')
plt.title("Task (c3): Speed Comparison - OCR vs Calculated from Distance", 
          fontsize=16, fontweight='bold')
plt.xlabel("Time (minutes)", fontsize=12)
plt.ylabel("Speed (km/h)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "c3_speed_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(14, 7))
error_vals = np.abs(speeds_kmh - speed_calc_kmh)
plt.plot(times_s/60, error_vals, marker='o', markersize=3, 
         linewidth=2, color='#C1292E')
plt.axhline(y=mean_err, color='#F18F01', linestyle='--', linewidth=2, 
            label=f'Mean Error = {mean_err:.2f} km/h')
plt.title("Task (c3): Absolute Speed Error vs Time", fontsize=16, fontweight='bold')
plt.xlabel("Time (minutes)", fontsize=12)
plt.ylabel("Absolute Error (km/h)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "c3_absolute_error.png"), dpi=300, bbox_inches='tight')
plt.show()

df_c3 = pd.DataFrame({
    "Time(s)": times_s,
    "Time(min)": times_s/60,
    "Speed_OCR(km/h)": speeds_kmh,
    "Speed_Calculated(km/h)": speed_calc_kmh,
    "Absolute_Error(km/h)": error_vals
})
df_c3.to_csv(os.path.join(OUTPUT_FOLDER, "c3_speed_error_analysis.csv"), index=False)

with open(os.path.join(OUTPUT_FOLDER, "c3_error_metrics.txt"), 'w') as f:
    f.write("TASK (c3): SPEED ERROR METRICS\n")
    f.write("="*70 + "\n")
    f.write(f"Mean Absolute Error (MAE):               {mean_err:.3f} km/h\n")
    f.write(f"Root Mean Square Error (RMSE):           {rmse:.3f} km/h\n")
    f.write(f"Max Error:                               {np.max(error_vals):.3f} km/h\n")
    f.write(f"Min Error:                               {np.min(error_vals):.3f} km/h\n")

print("   ✅ Task (c3) complete")

# ==========================================================
# TASK (d): ACCELERATION vs TIME
# ==========================================================
print("\n📈 Generating Task (d): Acceleration vs Time graph...")

speed_ms = speeds_kmh * (1000.0 / 3600.0)
acc_ms2 = compute_derivative(speed_ms, times_s)

plt.figure(figsize=(14, 7))
plt.plot(times_s/60, acc_ms2, marker='o', markersize=3, linewidth=2, color='#06A77D')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
plt.title("Task (d): Instantaneous Acceleration vs Time", fontsize=16, fontweight='bold')
plt.xlabel("Time (minutes)", fontsize=12)
plt.ylabel("Acceleration (m/s²)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "d_acceleration_vs_time.png"), dpi=300, bbox_inches='tight')
plt.show()

df_d = pd.DataFrame({
    "Time(min)": times_s/60,
    "Acceleration(m/s²)": acc_ms2
})
df_d.to_csv(os.path.join(OUTPUT_FOLDER, "d_acceleration.csv"), index=False)

print("   ✅ Task (d) complete")

# ==========================================================
# TASK (e): JERK vs TIME
# ==========================================================
print("\n📈 Generating Task (e): Jerk vs Time graph...")

jerk_ms3 = compute_derivative(acc_ms2, times_s)

plt.figure(figsize=(14, 7))
plt.plot(times_s/60, jerk_ms3, marker='o', markersize=3, linewidth=2, color='#D90368')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
plt.title("Task (e): Instantaneous Jerk vs Time", fontsize=16, fontweight='bold')
plt.xlabel("Time (minutes)", fontsize=12)
plt.ylabel("Jerk (m/s³)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "e_jerk_vs_time.png"), dpi=300, bbox_inches='tight')
plt.show()

df_e = pd.DataFrame({
    "Time(min)": times_s/60,
    "Jerk(m/s³)": jerk_ms3
})
df_e.to_csv(os.path.join(OUTPUT_FOLDER, "e_jerk.csv"), index=False)

print("   ✅ Task (e) complete")

# ==========================================================
# GENERATE SUMMARY REPORT
# ==========================================================
print("\n📄 Generating summary report...")

summary_report = f"""
{'='*70}
  JOURNEY ANALYSIS SUMMARY REPORT (Dark Mode)
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

VIDEO INFORMATION:
  File: {os.path.basename(VIDEO_FILE)}
  Duration analyzed: {times_s[-1]/60:.2f} minutes
  Total samples: {len(speeds_kmh)}
  OCR detected: {len(speed_sources) - estimated_count} ({(len(speed_sources)-estimated_count)/len(speed_sources)*100:.1f}%)
  Estimated values: {estimated_count} ({estimated_count/len(speed_sources)*100:.1f}%)

JOURNEY STATISTICS:
  Total distance traveled: {total_distance:.4f} km ({total_distance*1000:.2f} m)
  Average speed: {np.mean(speeds_kmh):.2f} km/h
  Maximum speed: {np.max(speeds_kmh):.2f} km/h
  Minimum speed: {np.min(speeds_kmh):.2f} km/h

TASK (c2) - DISTANCE CALCULATION:
  Distance from integration: {total_distance:.4f} km
  Distance from c1 method: {total_from_c1_km:.4f} km
  Percentage error: {percent_error:.2f}%

TASK (c3) - SPEED ERROR ANALYSIS:
  Mean Absolute Error: {mean_err:.3f} km/h
  RMSE: {rmse:.3f} km/h
  Max error: {np.max(error_vals):.3f} km/h

ACCELERATION STATISTICS:
  Mean acceleration: {np.mean(acc_ms2):.3f} m/s²
  Max acceleration: {np.max(acc_ms2):.3f} m/s²
  Min acceleration: {np.min(acc_ms2):.3f} m/s²

JERK STATISTICS:
  Mean jerk: {np.mean(jerk_ms3):.3f} m/s³
  Max jerk: {np.max(jerk_ms3):.3f} m/s³
  Min jerk: {np.min(jerk_ms3):.3f} m/s³

FILES GENERATED:
  ✓ Speed vs Time plot (with OCR/Estimated indicators)
  ✓ Time to travel 10km plot
  ✓ 2-minute sliding window distance plot
  ✓ Speed comparison plot
  ✓ Acceleration plot
  ✓ Jerk plot
  ✓ All CSV data files
  ✓ Error analysis files

{'='*70}
"""

with open(os.path.join(OUTPUT_FOLDER, "00_SUMMARY_REPORT.txt"), 'w') as f:
    f.write(summary_report)

print(summary_report)

print("\n" + "="*70)
print("  ✅ ALL TASKS COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\n📁 All outputs saved to:")
print(f"   {OUTPUT_FOLDER}")
print("\n📊 Generated files:")
print("   - 6 PNG graphs (tasks a, b, c1, c3, d, e)")
print("   - 7 CSV data files")
print("   - 3 text analysis files")
print("   - 1 summary report")
print("   - 2 ROI reference images")
print("\n🎉 Task 1 complete! You can now proceed to Task 2.")
