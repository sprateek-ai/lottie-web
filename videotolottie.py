#!/usr/bin/env python3
"""
Compact Video → Lottie Converter (CLI)
Optimized for:
- UI / cursor videos
- Small JSON size
- Smooth playback

Usage:
  python videotolottie_compact.py input.mp4
  python videotolottie_compact.py input.mp4 --fps 30 --quality 80
"""

import cv2
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import argparse
import os
import sys


class CompactVideoToLottie:
    def __init__(self, video_path, output_path, fps=30, quality=85, max_width=720):
        self.video_path = video_path
        self.output_path = output_path
        self.fps = min(fps, 30)
        self.quality = quality
        self.max_width = max_width

    # -----------------------------
    # FRAME EXTRACTION
    # -----------------------------
    def extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(orig_fps / self.fps))

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if w > self.max_width:
            scale = self.max_width / w
            w = self.max_width
            h = int(h * scale)

        frames = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % interval == 0:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            idx += 1

        cap.release()
        return frames, w, h

    # -----------------------------
    # TEMPORAL SMOOTHING
    # -----------------------------
    def smooth_frames(self, frames, alpha=0.7):
        smoothed = [frames[0]]
        for i in range(1, len(frames)):
            prev = smoothed[-1].astype(np.float32)
            curr = frames[i].astype(np.float32)
            blended = cv2.addWeighted(curr, alpha, prev, 1 - alpha, 0)
            smoothed.append(blended.astype(np.uint8))
        return smoothed

    # -----------------------------
    # COLOR QUANTIZATION
    # -----------------------------
    def quantize(self, frame, colors=128):
        img = Image.fromarray(frame)
        img = img.convert("P", palette=Image.ADAPTIVE, colors=colors)
        return np.array(img.convert("RGB"))

    # -----------------------------
    # WEBP ENCODING
    # -----------------------------
    def encode_webp(self, frame):
        img = Image.fromarray(frame)
        buf = BytesIO()

        q = self.quality
        if self.fps >= 30:
            q = min(q, 75)

        img.save(
            buf,
            format="WEBP",
            quality=q,
            method=6,
            lossless=False
        )

        return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()

    # -----------------------------
    # DEDUPLICATION
    # -----------------------------
    def deduplicate(self, frames):
        assets = []
        frame_refs = []
        cache = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            motion = np.mean(edges)

            matched = False
            for i, (g, _) in enumerate(cache):
                mse = np.mean((gray.astype(float) - g.astype(float)) ** 2)
                similarity = 1.0 - (np.sqrt(mse) / 255.0)

                if similarity > 0.999 and motion < 2:
                    frame_refs.append(i)
                    matched = True
                    break

            if not matched:
                qf = self.quantize(frame)
                encoded = self.encode_webp(qf)
                asset_id = len(assets)

                assets.append({
                    "id": f"img_{asset_id}",
                    "w": frame.shape[1],
                    "h": frame.shape[0],
                    "p": encoded,
                    "e": 1
                })

                cache.append((gray, frame))
                frame_refs.append(asset_id)

        return assets, frame_refs

    # -----------------------------
    # LOTTIE BUILD
    # -----------------------------
    def build_lottie(self, assets, frame_refs, w, h):
        layers = []
        hold = max(2, int(self.fps / 15))

        for i, ref in enumerate(frame_refs):
            layers.append({
                "ty": 2,
                "refId": f"img_{ref}",
                "ks": {
                    "o": {"k": 100},
                    "r": {"k": 0},
                    "p": {"k": [w / 2, h / 2, 0]},
                    "a": {"k": [w / 2, h / 2, 0]},
                    "s": {"k": [100, 100, 100]}
                },
                "ip": i,
                "op": i + hold
            })

        return {
            "v": "5.7.4",
            "fr": self.fps,
            "ip": 0,
            "op": len(frame_refs),
            "w": w,
            "h": h,
            "assets": assets,
            "layers": layers
        }

    # -----------------------------
    # MAIN CONVERT
    # -----------------------------
    def convert(self):
        frames, w, h = self.extract_frames()
        frames = self.smooth_frames(frames)
        assets, refs = self.deduplicate(frames)
        lottie = self.build_lottie(assets, refs, w, h)

        with open(self.output_path, "w") as f:
            json.dump(lottie, f, separators=(",", ":"))

        return self.output_path


# ==========================================================
# CLI ENTRY POINT
# ==========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Convert video to compact Lottie JSON"
    )

    parser.add_argument("video", help="Input video file (mp4, webm, etc)")
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file (default: <input>_lottie.json)"
    )
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (max 30)")
    parser.add_argument("--quality", type=int, default=85, help="WebP quality 1–100")
    parser.add_argument("--max-width", type=int, default=720, help="Max output width")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ File not found: {args.video}")
        sys.exit(1)

    output = args.output
    if not output:
        base = os.path.splitext(args.video)[0]
        output = base + "_lottie.json"

    converter = CompactVideoToLottie(
        video_path=args.video,
        output_path=output,
        fps=args.fps,
        quality=args.quality,
        max_width=args.max_width
    )

    print("▶ Converting:", args.video)
    result = converter.convert()

    size_mb = os.path.getsize(result) / (1024 * 1024)
    print("✅ Done")
    print("📄 Output:", result)
    print(f"📦 Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
