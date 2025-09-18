import matplotlib.pyplot as plt
import cv2
import os
from os.path import join, basename
from collections import deque
from lane_detection import color_frame_pipeline

if __name__ == '__main__':
    resize_h, resize_w = 540, 960

    verbose = True
    if verbose:
        plt.ion()
        figManager = plt.get_current_fig_manager()
        try:
            figManager.window.wm_state('zoomed')  # Maximize window safely
        except AttributeError:
            print("Warning: Unable to maximize Matplotlib window on this platform.")

    # Ensure output directories exist
    os.makedirs(join('out', 'images'), exist_ok=True)
    os.makedirs(join('out', 'videos'), exist_ok=True)

    # Test on images
    test_images_dir = join('data', 'test_images')
    test_images = [join(test_images_dir, name) for name in os.listdir(test_images_dir)]

    for test_img in test_images:
        print(f'Processing image: {test_img}')

        out_path = join('out', 'images', basename(test_img))
        in_image = cv2.imread(test_img, cv2.IMREAD_COLOR)

        if in_image is None:
            print(f"Error: Unable to read image {test_img}, skipping...")
            continue

        in_image = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)
        out_image = color_frame_pipeline([in_image], solid_lines=True)

        if out_image is None:
            print(f"Error: Processing failed for {test_img}, skipping...")
            continue

        cv2.imwrite(out_path, cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR))

        if verbose:
            plt.imshow(out_image)
            plt.waitforbuttonpress()

    plt.close('all')

    # Test on videos
    test_videos_dir = join('data', 'test_videos')
    test_videos = [join(test_videos_dir, name) for name in os.listdir(test_videos_dir)]

    for test_video in test_videos:
        print(f'Processing video: {test_video}')

        cap = cv2.VideoCapture(test_video)

        if not cap.isOpened():
            print(f"Error: Unable to open video {test_video}, skipping...")
            continue

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out_path = join('out', 'videos', basename(test_video))
        out = cv2.VideoWriter(out_path, fourcc, 20.0, (resize_w, resize_h))

        frame_buffer = deque(maxlen=10)
        while cap.isOpened():
            ret, color_frame = cap.read()
            if not ret:
                break

            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            color_frame = cv2.resize(color_frame, (resize_w, resize_h))
            frame_buffer.append(color_frame)

            blend_frame = color_frame_pipeline(frames=frame_buffer, solid_lines=True, temporal_smoothing=True)

            if blend_frame is None:
                print("Warning: Skipping invalid processed frame.")
                continue

            out.write(cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR))

            cv2.imshow('blend', cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit early
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
