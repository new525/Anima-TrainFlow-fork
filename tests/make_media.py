"""Make test media where frame i is solid gray (10 + 4*i), so the extracted frame's index is readable from its color."""
import subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
import imageio_ffmpeg
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
FF = imageio_ffmpeg.get_ffmpeg_exe()
N, W, H = 61, 320, 180            # 61 frames -> middle index 30 -> gray 130
def frames(w=W, h=H):
    for i in range(N):
        f = np.full((h, w, 3), 10 + 4 * i, np.uint8)
        f[:20, :40] = (255, 0, 0)  # red marker top-left: shows orientation
        yield f
def encode(name, args, w=W, h=H):
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "24", "-i", "-",
                          *args, str(out / name)], stdin=subprocess.PIPE)
    for f in frames(w, h): p.stdin.write(f.tobytes())
    p.stdin.close(); assert p.wait() == 0, name
encode("clip_h264.mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "12"])
encode("clip_vp9.webm", ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30", "-g", "12"])
encode("clip_h264.mkv", ["-c:v", "libx264", "-pix_fmt", "yuv420p"])
encode("clip_mjpeg.avi", ["-c:v", "mjpeg", "-q:v", "3"])
encode("clip.mov", ["-c:v", "libx264", "-pix_fmt", "yuv420p"])
# phone-style: landscape pixels + "rotate 90" display matrix
subprocess.run([FF, "-y", "-loglevel", "error", "-display_rotation:v:0", "-90", "-i", str(out / "clip.mov"), "-c", "copy",
                str(out / "phone_rotated.mp4")], check=True)
gif = [Image.fromarray(f) for f in frames()]
gif[0].save(out / "anim.gif", save_all=True, append_images=gif[1:], duration=40, loop=0)
Image.fromarray(next(frames())).save(out / "still.gif")
a = np.zeros((300, 200, 4), np.uint8); a[..., :3] = (20, 120, 220); a[..., 3] = 255; a[:, :50, 3] = 0   # left strip transparent
Image.fromarray(a, "RGBA").save(out / "photo.avif", quality=90)
(out / "broken.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42 not really a video")
print(sorted(p.name for p in out.iterdir()))
