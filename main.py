import cv2
import numpy as np
import json
import os
import base64
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class PixelPerfectVideoToLottie:
    def __init__(self,
                 fps_target: int = None,
                 scale: float = 1.0,
                 quality: int = 85):
        """
        Initialize converter that preserves video exactly as-is
       
        Args:
            fps_target: Target FPS (None = use original)
            scale: Scale factor (1.0 = original size)
            quality: JPEG quality for embedded images (1-100)
        """
        self.fps_target = fps_target
        self.scale = scale
        self.quality = quality
       
    def extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], float, Tuple[int, int]]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
       
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
       
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       
        # Calculate new dimensions
        new_width = int(width * self.scale)
        new_height = int(height * self.scale)
       
        # Determine frame skip
        if self.fps_target:
            frame_skip = max(1, int(fps / self.fps_target))
            output_fps = fps / frame_skip
        else:
            frame_skip = 1
            output_fps = fps
       
        print(f"📹 Input: {width}x{height} @ {fps:.1f}fps ({total_frames} frames)")
        print(f"📤 Output: {new_width}x{new_height} @ {output_fps:.1f}fps")
        print(f"⏭️  Frame skip: {frame_skip}")
       
        frames = []
        frame_idx = 0
       
        while True:
            ret, frame = cap.read()
            if not ret:
                break
           
            if frame_idx % frame_skip == 0:
                # Resize if needed
                if self.scale != 1.0:
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
               
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
           
            frame_idx += 1
           
            if frame_idx % 50 == 0:
                print(f"  Reading: {frame_idx}/{total_frames} ({len(frames)} kept)")
       
        cap.release()
        print(f"✅ Extracted {len(frames)} frames")
       
        return frames, output_fps, (new_width, new_height)
   
    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 JPEG"""
        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), encode_param)
       
        # Convert to base64
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"
   
    def create_image_asset(self, frame: np.ndarray, asset_id: str, width: int, height: int) -> Dict:
        """Create Lottie image asset"""
        base64_data = self.frame_to_base64(frame)
       
        return {
            'id': asset_id,
            'w': width,
            'h': height,
            'u': '',
            'p': base64_data,
            'e': 0
        }
   
    def create_lottie_json(self,
                          frames: List[np.ndarray],
                          fps: float,
                          dimensions: Tuple[int, int]) -> Dict:
        """Create Lottie JSON with embedded images"""
        width, height = dimensions
        total_frames = len(frames)
       
        print(f"\n🔨 Building Lottie JSON...")
        print(f"  📊 {total_frames} frames @ {fps:.1f}fps")
       
        # Create assets (images)
        assets = []
        for i, frame in enumerate(frames):
            asset = self.create_image_asset(frame, f"img_{i}", width, height)
            assets.append(asset)
           
            if (i + 1) % 10 == 0 or i == total_frames - 1:
                print(f"  Encoding: {i+1}/{total_frames} frames...")
       
        # Create image layers with time remapping
        layers = []
       
        for i in range(total_frames):
            layer = {
                'ddd': 0,
                'ind': i + 1,
                'ty': 2,  # Image layer
                'nm': f'Frame {i}',
                'refId': f'img_{i}',
                'sr': 1,
                'ks': {
                    'o': {'a': 0, 'k': 100},
                    'r': {'a': 0, 'k': 0},
                    'p': {'a': 0, 'k': [width/2, height/2, 0]},
                    'a': {'a': 0, 'k': [width/2, height/2, 0]},
                    's': {'a': 0, 'k': [100, 100, 100]}
                },
                'ao': 0,
                'ip': i,
                'op': i + 1,
                'st': 0,
                'bm': 0
            }
            layers.append(layer)
       
        # Build main Lottie structure
        lottie_data = {
            'v': '5.9.0',
            'fr': fps,
            'ip': 0,
            'op': total_frames,
            'w': width,
            'h': height,
            'nm': 'Video Animation',
            'ddd': 0,
            'assets': assets,
            'layers': layers,
            'markers': []
        }
       
        print(f"✅ Created {len(layers)} image layers")
        return lottie_data
   
    def save_json(self, lottie_data: Dict, output_path: str, minify: bool = True):
        """Save Lottie JSON (minified single line)"""
        print(f"\n💾 Saving to {output_path}...")
       
        if minify:
            # Single line, no spaces
            with open(output_path, 'w') as f:
                json.dump(lottie_data, f, separators=(',', ':'), ensure_ascii=False)
        else:
            # Pretty print
            with open(output_path, 'w') as f:
                json.dump(lottie_data, f, indent=2, ensure_ascii=False)
       
        file_size = os.path.getsize(output_path)
        print(f"✅ Saved: {file_size / (1024*1024):.2f} MB")
       
        if file_size > 10 * 1024 * 1024:  # > 10MB
            print(f"\n⚠️  Large file size! To reduce:")
            print(f"   • Lower quality (current: {self.quality})")
            print(f"   • Reduce scale (current: {self.scale})")
            print(f"   • Lower fps_target (current: {self.fps_target})")
   
    def convert(self, video_path: str, output_path: str = None) -> str:
        """Main conversion pipeline - preserves exact video appearance"""
        if output_path is None:
            output_path = Path(video_path).stem + '_lottie.json'
       
        print("=" * 70)
        print("🎬 PIXEL-PERFECT VIDEO TO LOTTIE CONVERTER")
        print("   (Preserves exact video appearance)")
        print("=" * 70)
       
        # Extract frames
        print("\n[1/2] 📹 Extracting frames...")
        frames, fps, dimensions = self.extract_frames(video_path)
       
        # Create Lottie JSON
        print("\n[2/2] 🎨 Creating Lottie JSON...")
        lottie_data = self.create_lottie_json(frames, fps, dimensions)
       
        # Save
        self.save_json(lottie_data, output_path, minify=True)
       
        print("\n" + "=" * 70)
        print("✅ CONVERSION COMPLETE!")
        print("=" * 70)
        print(f"📁 Output: {output_path}")
        print(f"⏱️  Duration: {len(frames)/fps:.2f}s")
        print(f"🎬 Frames: {len(frames)}")
        print(f"📐 Size: {dimensions[0]}x{dimensions[1]}")
       
        # Usage instructions
        print("\n📖 WEB USAGE:")
        print(f"""
<!-- Using dotLottie Player -->
<script src="https://unpkg.com/@dotlottie/player-component@latest/dist/dotlottie-player.mjs" type="module"></script>
<dotlottie-player
    src="{output_path}"
    background="transparent"
    speed="1"
    style="width: 640px; height: 360px"
    loop
    autoplay>
</dotlottie-player>


<!-- OR Lottie Web -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/bodymovin/5.12.2/lottie.min.js"></script>
<div id="lottie"></div>
<script>
lottie.loadAnimation({{
  container: document.getElementById('lottie'),
  renderer: 'svg',
  loop: true,
  autoplay: true,
  path: '{output_path}'
}});
</script>


<!-- Preview at: https://lottiefiles.com/preview -->
        """)
       
        return output_path




class OptimizedVideoToLottie(PixelPerfectVideoToLottie):
    """Optimized version for smaller file sizes"""
   
    def __init__(self, preset: str = 'balanced'):
        """
        Preset-based initialization
       
        Presets:
            'web-small': 480p, 15fps, quality 75 (~5-10MB)
            'balanced': 720p, 20fps, quality 80 (~10-20MB)
            'high-quality': 1080p, 25fps, quality 90 (~30-50MB)
            'original': No scaling, original fps, quality 95
        """
        presets = {
            'web-small': {'fps': 15, 'scale': 0.5, 'quality': 75},
            'balanced': {'fps': 20, 'scale': 0.7, 'quality': 80},
            'high-quality': {'fps': 25, 'scale': 0.9, 'quality': 90},
            'original': {'fps': None, 'scale': 1.0, 'quality': 95}
        }
       
        config = presets.get(preset, presets['balanced'])
        super().__init__(
            fps_target=config['fps'],
            scale=config['scale'],
            quality=config['quality']
        )
        print(f"⚙️  Preset: {preset.upper()}")




def main():
    """CLI Interface"""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     PIXEL-PERFECT VIDEO TO LOTTIE CONVERTER                   ║
    ║        Preserves Exact Video Appearance (Image-Based)         ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
   
    video_path = input("📂 Enter video file path: ").strip()
   
    if not os.path.exists(video_path):
        print("❌ Video file not found!")
        return
   
    print("\n🎯 Select Preset:")
    print("  1. Web Small     (480p, 15fps, ~5-10MB)")
    print("  2. Balanced      (720p, 20fps, ~10-20MB) ⭐ RECOMMENDED")
    print("  3. High Quality  (1080p, 25fps, ~30-50MB)")
    print("  4. Original      (Full res, original fps, ~50-150MB)")
    print("  5. Custom        (Set your own parameters)")
   
    choice = input("\nChoice [2]: ").strip() or '2'
   
    if choice == '5':
        # Custom settings
        fps = input("Target FPS (blank for original): ").strip()
        fps = int(fps) if fps else None
       
        scale = input("Scale factor (1.0 = original) [0.7]: ").strip()
        scale = float(scale) if scale else 0.7
       
        quality = input("JPEG quality (1-100) [80]: ").strip()
        quality = int(quality) if quality else 80
       
        converter = PixelPerfectVideoToLottie(
            fps_target=fps,
            scale=scale,
            quality=quality
        )
    else:
        # Use preset
        preset_map = {
            '1': 'web-small',
            '2': 'balanced',
            '3': 'high-quality',
            '4': 'original'
        }
        preset = preset_map.get(choice, 'balanced')
        converter = OptimizedVideoToLottie(preset=preset)
   
    # Convert
    try:
        output_path = converter.convert(video_path)
        print(f"\n🎉 SUCCESS! Your Lottie animation is ready!")
        print(f"\n📦 File: {output_path}")
        print(f"\n💡 Preview at: https://lottiefiles.com/preview")
        print(f"   (Upload your JSON file to preview)")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()




if __name__ == "__main__":
    main()
