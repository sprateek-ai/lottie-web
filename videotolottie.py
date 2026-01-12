#!/usr/bin/env python3
"""
Video to Lottie JSON Converter
Converts video files into Lottie animation JSON format with base64 encoded frames
Optimized for smaller file sizes while maintaining quality
"""

import cv2
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import argparse
import os


class VideoToLottieConverter:
    def __init__(self, video_path, output_path=None, fps=15, quality=85, max_width=608):
        """
        Initialize the converter
        
        Args:
            video_path: Path to input video file
            output_path: Path to output JSON file
            fps: Target frames per second
            quality: WebP quality (1-100)
            max_width: Maximum width for output frames
        """
        self.video_path = video_path
        self.output_path = output_path or video_path.rsplit('.', 1)[0] + '_lottie.json'
        self.target_fps = fps
        self.quality = quality
        self.max_width = max_width
        
    def extract_frames(self):
        """Extract frames from video at specified FPS"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties:")
        print(f"  Original FPS: {original_fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Resolution: {width}x{height}")
        
        # Calculate frame sampling rate
        frame_interval = max(1, int(original_fps / self.target_fps))
        
        # Calculate output dimensions maintaining aspect ratio
        if width > self.max_width:
            output_width = self.max_width
            output_height = int(height * (self.max_width / width))
        else:
            output_width = width
            output_height = height
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        print(f"Extracting frames (every {frame_interval} frame(s))...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Resize with high-quality interpolation
                resized = cv2.resize(frame, (output_width, output_height), 
                                    interpolation=cv2.INTER_LANCZOS4)
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                extracted_count += 1
                
                if extracted_count % 10 == 0:
                    print(f"  Extracted {extracted_count} frames...")
            
            frame_count += 1
        
        cap.release()
        
        print(f"Total frames extracted: {extracted_count}")
        return frames, output_width, output_height
    
    def detect_duplicate_frames(self, frames, threshold=0.98):
        """Detect duplicate or near-duplicate frames to optimize file size"""
        print("Analyzing frames for duplicates...")
        
        frame_map = {}  # Maps frame index to unique image ID
        unique_frames = []
        unique_ids = []
        
        for idx, frame in enumerate(frames):
            # Convert to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Check similarity with existing unique frames
            is_duplicate = False
            for unique_idx, (unique_gray, unique_frame) in enumerate(unique_frames):
                # Calculate structural similarity
                similarity = self._calculate_similarity(gray, unique_gray)
                
                if similarity >= threshold:
                    frame_map[idx] = unique_ids[unique_idx]
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_id = len(unique_frames)
                unique_frames.append((gray, frame))
                unique_ids.append(unique_id)
                frame_map[idx] = unique_id
        
        print(f"  Found {len(unique_frames)} unique frames out of {len(frames)}")
        print(f"  Space savings: {((1 - len(unique_frames)/len(frames)) * 100):.1f}%")
        
        return frame_map, [f for _, f in unique_frames]
    
    def _calculate_similarity(self, img1, img2):
        """Calculate similarity between two grayscale images"""
        if img1.shape != img2.shape:
            return 0.0
        
        # Compute mean squared error
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        
        # Convert to similarity score (1 = identical, 0 = completely different)
        max_pixel_value = 255.0
        similarity = 1.0 - (np.sqrt(mse) / max_pixel_value)
        
        return similarity
    
    def frame_to_base64_webp(self, frame):
        """Convert numpy frame to optimized base64 WebP"""
        pil_image = Image.fromarray(frame)
        
        buffer = BytesIO()
        
        # WebP with optimization
        pil_image.save(
            buffer, 
            format='WEBP',
            quality=self.quality,
            method=6,  # Maximum compression effort
            lossless=False
        )
        
        webp_data = buffer.getvalue()
        base64_data = base64.b64encode(webp_data).decode('utf-8')
        
        return f"data:image/webp;base64,{base64_data}"
    
    def create_lottie_json(self, frames, width, height):
        """Create Lottie JSON structure matching the exact format"""
        num_frames = len(frames)
        
        # Detect duplicate frames for optimization
        frame_map, unique_frames = self.detect_duplicate_frames(frames)
        
        print("Converting frames to base64 WebP format...")
        
        # Create assets array with unique frames only
        assets = []
        for idx, frame in enumerate(unique_frames):
            if (idx + 1) % 5 == 0 or idx == 0:
                print(f"  Processing unique frame {idx + 1}/{len(unique_frames)}...")
            
            base64_image = self.frame_to_base64_webp(frame)
            
            asset = {
                "id": f"image_{idx}",
                "w": width,
                "h": height,
                "p": base64_image,
                "e": 1
            }
            assets.append(asset)
        
        # Create layers - one per frame, referencing appropriate asset
        layers = []
        for idx in range(num_frames):
            unique_id = frame_map[idx]
            
            layer = {
                "ddd": 0,
                "ind": idx + 1,
                "ty": 2,
                "refId": f"image_{unique_id}",
                "ks": {
                    "o": {"k": 100},
                    "r": {"k": 0},
                    "p": {"k": [width/2, height/2, 0]},
                    "a": {"k": [width/2, height/2, 0]},
                    "s": {"k": [100, 100, 100]}
                },
                "ao": 0,
                "ip": idx,
                "op": idx + 1,
                "st": idx,
                "bm": 0
            }
            layers.append(layer)
        
        # Create main Lottie structure (exact format match)
        lottie_json = {
            "v": "5.7.4",
            "fr": self.target_fps,
            "ip": 0,
            "op": num_frames,
            "w": width,
            "h": height,
            "assets": assets,
            "layers": layers
        }
        
        return lottie_json
    
    def convert(self):
        """Main conversion process"""
        print(f"\n{'='*60}")
        print(f"Video to Lottie Converter")
        print(f"{'='*60}\n")
        print(f"Input: {self.video_path}")
        print(f"Output: {self.output_path}")
        print(f"Target FPS: {self.target_fps}")
        print(f"Quality: {self.quality}")
        print(f"Max Width: {self.max_width}\n")
        
        # Extract frames
        frames, width, height = self.extract_frames()
        
        # Create Lottie JSON
        lottie_data = self.create_lottie_json(frames, width, height)
        
        # Save to file with minimal whitespace
        print(f"\nSaving Lottie JSON to: {self.output_path}")
        with open(self.output_path, 'w') as f:
            json.dump(lottie_data, f, separators=(',', ':'))
        
        file_size = os.path.getsize(self.output_path)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"\n{'='*60}")
        print(f"✅ Conversion Complete!")
        print(f"{'='*60}")
        print(f"Output file size: {file_size_mb:.2f} MB")
        print(f"Total frames: {len(frames)}")
        print(f"Unique frames: {len(lottie_data['assets'])}")
        print(f"Frame rate: {self.target_fps} FPS")
        print(f"Resolution: {width}x{height}")
        print(f"{'='*60}\n")
        
        return self.output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert video files to Lottie JSON format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_to_lottie.py input.mp4
  python video_to_lottie.py input.mp4 -o output.json
  python video_to_lottie.py input.mp4 --fps 30 --quality 90
  python video_to_lottie.py input.mp4 --max-width 800
        """
    )
    
    parser.add_argument('video', help='Input video file path')
    parser.add_argument('-o', '--output', help='Output JSON file path')
    parser.add_argument('--fps', type=int, default=15, 
                       help='Target frames per second (default: 15)')
    parser.add_argument('--quality', type=int, default=85,
                       help='WebP quality 1-100 (default: 85)')
    parser.add_argument('--max-width', type=int, default=608,
                       help='Maximum output width (default: 608)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        return
    
    if args.quality < 1 or args.quality > 100:
        print("❌ Error: Quality must be between 1 and 100")
        return
    
    if args.fps < 1 or args.fps > 60:
        print("❌ Error: FPS must be between 1 and 60")
        return
    
    # Create converter and run
    converter = VideoToLottieConverter(
        video_path=args.video,
        output_path=args.output,
        fps=args.fps,
        quality=args.quality,
        max_width=args.max_width
    )
    
    try:
        output_file = converter.convert()
        print(f"✅ Success! Lottie JSON saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        raise


if __name__ == "__main__":
    main()