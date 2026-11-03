import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(ouput_video_frames,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()


def stream_draw_save(video_path, draw_fn, output_video_path, fps=25):
    """Re-decode the source frame-by-frame, draw on each via draw_fn(frame, i),
    write it, and discard it. Never holds the whole (input or annotated) video
    in RAM — that's the 2x peak the eager read_video/draw/save path costs.
    Decode is sequential, so frame i lines up with tracks[i].
    mp4v/.mp4 because QuickTime stutters on XVID/.avi; fps = the source's."""
    cap = cv2.VideoCapture(video_path)
    out = None
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = draw_fn(frame, i)
        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps,
                                  (frame.shape[1], frame.shape[0]))
        out.write(frame)
        i += 1
    cap.release()
    if out is not None:
        out.release()
