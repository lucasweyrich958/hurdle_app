#Extract Frames with cv and label each frame with labelimg

import cv2
import os

def extract_frames_from_videos(videos_dir, output_dir, frame_rate=1):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for video_file in video_files:
        video_path = os.path.join(videos_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_rate == 0:
                frame_name = os.path.join(output_dir, f"{video_file}_frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_name, frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {saved_count} frames from {video_file}")

# Example usage
videos_dir = '/Users/Lucas/Documents/Hurdle App/Races'
output_dir = '/Users/Lucas/Documents/Hurdle App/Frames'
extract_frames_from_videos(videos_dir, output_dir, frame_rate=5)