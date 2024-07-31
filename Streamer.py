import subprocess

def get_streamer(width, height):
    # Initialize camera
    size_of_frame = f"{width}x{height}"

    # Define FFmpeg command with scaling
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        size_of_frame,  # Size of one frame
        "-r",
        "30",  # Frames per second
        "-i",
        "-",  # The input comes from a pipe
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        "-f",
        "flv",
        "rtmps://3cc909883bdc.global-contribute.live-video.net:443/app/sk_us-east-1_3eMAC654dfPi_s8NCm6DGByeOGS08yq2iIxTCMvR7KK",  # Replace with your IVS Ingest endpoint and Stream Key
    ]

    # Open pipe to FFmpeg
    return subprocess.Popen(command, stdin=subprocess.PIPE)
       
