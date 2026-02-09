import cv2
import random
import os

# ======================== CONFIGURATION ========================
VIDEO_PATH = r"C:\Users\akhilesh zende\Downloads\8064258-hd_1920_1080_24fps.mp4"
OUTPUT_PATH = r"C:\Users\akhilesh zende\Downloads\output_assignment0.mp4"

# Text Configuration
RANDOM_TEXT = "Kamna Tapase - MTech Automotive"
RIBBON_TEXT = "Automotive python assignment, version 0"

# ======================== VIDEO PROCESSING ========================
def process_video_with_text():
    """
    Process video to add random text and bottom ribbon
    """
    # Open video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video at {VIDEO_PATH}")
        print("Please check if the file exists and path is correct.")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video Properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.2f} seconds")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("❌ Error: Could not create output video writer")
        cap.release()
        return False
    
    # Initialize variables
    frame_count = 0
    rand_x, rand_y = 50, 100
    change_interval = int(fps * 1)  # Change position every 1 second
    
    # Ribbon configuration
    ribbon_height = 60
    ribbon_color = (0, 0, 0)  # Black background
    text_color = (255, 255, 255)  # White text
    random_text_color = (0, 215, 255)  # Gold/Yellow color (BGR format)
    
    print("\n🎬 Processing video...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Update random text position periodically
        if frame_count % change_interval == 0:
            # Calculate text size for proper boundary checking
            (text_width, text_height), baseline = cv2.getTextSize(
                RANDOM_TEXT,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                2
            )
            
            # Generate random position ensuring text stays within frame
            # Leave space for ribbon at bottom
            rand_x = random.randint(10, max(10, width - text_width - 10))
            rand_y = random.randint(text_height + 10, 
                                   max(text_height + 20, height - ribbon_height - 20))
        
        # ============ ADD RANDOM TEXT ============
        cv2.putText(frame,
                    RANDOM_TEXT,
                    (rand_x, rand_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    random_text_color,
                    3,
                    cv2.LINE_AA)
        
        # ============ ADD BOTTOM RIBBON ============
        # Draw black rectangle at bottom
        cv2.rectangle(frame,
                     (0, height - ribbon_height),
                     (width, height),
                     ribbon_color,
                     -1)  # -1 fills the rectangle
        
        # Add text to ribbon (centered)
        (ribbon_text_width, ribbon_text_height), _ = cv2.getTextSize(
            RIBBON_TEXT,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            2
        )
        
        ribbon_text_x = (width - ribbon_text_width) // 2
        ribbon_text_y = height - (ribbon_height - ribbon_text_height) // 2
        
        cv2.putText(frame,
                    RIBBON_TEXT,
                    (ribbon_text_x, ribbon_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    text_color,
                    2,
                    cv2.LINE_AA)
        
        # Write frame to output video
        out.write(frame)
        
        # Progress indicator
        if frame_count % int(fps * 5) == 0:  # Every 5 seconds
            progress = (frame_count / total_frames) * 100
            print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        frame_count += 1
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"\n✅ Video processing complete!")
    print(f"   Total frames processed: {frame_count}")
    print(f"   Output saved to: {OUTPUT_PATH}")
    
    # Verify output file exists
    if os.path.exists(OUTPUT_PATH):
        output_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)  # Size in MB
        print(f"   Output file size: {output_size:.2f} MB")
        return True
    else:
        print("❌ Warning: Output file was not created successfully")
        return False

# ======================== MAIN EXECUTION ========================
if __name__ == "__main__":
    print("="*60)
    print("  PYTHON ASSIGNMENT - TASK 0")
    print("  Video Processing with Random Text and Ribbon")
    print("="*60)
    
    # Check if input video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"\n❌ Error: Input video not found at:")
        print(f"   {VIDEO_PATH}")
        print("\nPlease check the path and try again.")
    else:
        success = process_video_with_text()
        
        if success:
            print("\n" + "="*60)
            print("  ✅ TASK 0 COMPLETED SUCCESSFULLY!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("  ❌ TASK 0 FAILED")
            print("="*60)
