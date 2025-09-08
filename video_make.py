import cv2
import os

# Path to your folder of images
image_folder = "/home/rizo/mipt_ccm/bbq/warehouse/results"
video_name = "/home/rizo/mipt_ccm/bbq/warehouse/output_video.mp4"
fps = 30  # Change this to your desired FPS

# Get list of images
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort()  # Sort by filename

# Read the first image to get dimensions
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# Write images to video
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

video.release()
print(f"Video saved as {video_name}")
