import whisper
import csv
import os
import torch
import subprocess
import json

# =========================
# CONFIG
# =========================
VIDEO_DIR = "videos"
CSV_DIR = "csv"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov")

os.makedirs(CSV_DIR, exist_ok=True)

# =========================
# HELPERS
# =========================
def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def get_video_duration(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])

# =========================
# GPU CHECK
# =========================
assert torch.cuda.is_available(), "CUDA is not available"
print("Using GPU:", torch.cuda.get_device_name(0))

# =========================
# LOAD MODEL ONCE
# =========================
model = whisper.load_model("medium").to("cuda")

# =========================
# COLLECT VIDEOS
# =========================
video_files = sorted([
    os.path.join(VIDEO_DIR, f)
    for f in os.listdir(VIDEO_DIR)
    if f.lower().endswith(VIDEO_EXTENSIONS)
])

assert video_files, "No video files found"
print(f"Found {len(video_files)} videos")

# =========================
# PROCESS EACH VIDEO
# =========================
for video_path in video_files:
    video_name = os.path.basename(video_path)
    video_stem = os.path.splitext(video_name)[0]
    csv_path = os.path.join(CSV_DIR, f"{video_stem}.csv")

    print(f"\nTranscribing: {video_name}")

    duration_sec = get_video_duration(video_path)
    duration_mmss = format_time(duration_sec)

    result = model.transcribe(
        video_path,
        fp16=True,
        temperature=0.2,
        beam_size=5,
        best_of=5,
        condition_on_previous_text=False,
        initial_prompt="هذا درس يحتوي على كلمات عربية وفرنسية"
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Header
        writer.writerow([
            "video_duration_mmss",
            "segment_start_mmss",
            "segment_end_mmss",
            "text"
        ])

        for seg in result["segments"]:
            writer.writerow([
                duration_mmss,
                format_time(seg["start"]),
                format_time(seg["end"]),
                seg["text"]
            ])

    print(f"Saved → {csv_path}")

print("\nAll videos processed successfully 🎉")
