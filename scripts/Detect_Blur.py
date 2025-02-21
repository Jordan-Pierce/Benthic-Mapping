import blur_detector
import argparse
import cv2
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

class VideoBlurDetector:
    def __init__(self, video_path, every_n_frames, output_dir):
        self.video_path = video_path
        self.every_n_frames = every_n_frames
        self.output_dir = output_dir

    def is_not_blurry(self, frame):

        blur_map = blur_detector.detectBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                            downsampling_factor=4,
                                            num_scales=4,
                                            scale_start=2,
                                            num_iterations_RF_filter=3)

        blur_threshold = 0.40  # This value may need to be adjusted
        return blur_map.mean() > blur_threshold

    def process_frame(self, frame, frame_index, output_subdir):
        if self.is_not_blurry(frame):
            output_path = os.path.join(output_subdir, f"frame_{frame_index:04d}.jpg")
            cv2.imwrite(output_path, frame)
            return True
        return False

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        saved_frame_count = 0

        video_basename = os.path.basename(self.video_path)
        output_subdir = os.path.join(self.output_dir, os.path.splitext(video_basename)[0])
        os.makedirs(output_subdir, exist_ok=True)

        with ThreadPoolExecutor() as executor:
            futures = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.every_n_frames == 0:
                    futures.append(executor.submit(self.process_frame, frame, saved_frame_count, output_subdir))
                    saved_frame_count += 1

                frame_count += 1

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred while processing a frame: {e}")

        cap.release()

def main():
    parser = argparse.ArgumentParser(description="Detect non-blurry frames from a video file.")

    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the video file")
    parser.add_argument("--every_n_frames", type=int, required=True,
                        help="Process every n-th frame")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save non-blurry frames")

    args = parser.parse_args()

    try:
        detector = VideoBlurDetector(args.video_path, args.every_n_frames, args.output_dir)
        detector.process_video()
        print("Processing completed.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()