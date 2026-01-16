#!/usr/bin/env python3
"""
Flask Web Application for Video to Lottie Converter
Provides RESTful API with real-time progress updates
Fixed to handle AV1 and other codec issues
"""

from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import tempfile
import shutil
from pathlib import Path
import threading
import time
from videotolottie import CompactVideoToLottie
import cv2
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import subprocess

app = Flask(__name__, static_folder='static', template_folder='.')
CORS(app)

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'webm', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Store conversion progress
conversion_status = {}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_video(input_path, output_path, tracker=None):
    """
    Convert video to H.264/MP4 format that OpenCV can reliably handle.
    This fixes AV1 and other codec compatibility issues.
    """
    if tracker:
        tracker.update("preprocessing", 2, "Converting video to compatible format...")
    
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',      # Use H.264 codec (widely supported)
        '-preset', 'fast',       # Fast encoding
        '-crf', '23',            # Good quality
        '-pix_fmt', 'yuv420p',   # Ensure compatibility
        '-c:a', 'aac',           # Audio codec
        '-b:a', '128k',          # Audio bitrate
        '-movflags', '+faststart', # Web optimization
        '-y',                    # Overwrite output
        output_path
    ]
    
    try:
        # Run FFmpeg with output suppression
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True,
            timeout=300  # 5 minute timeout
        )
        return True
    except subprocess.TimeoutExpired:
        raise RuntimeError("Video conversion timed out (max 5 minutes)")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else "Unknown FFmpeg error"
        raise RuntimeError(f"Video conversion failed: {error_msg}")
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg: "
            "https://ffmpeg.org/download.html"
        )


class ProgressTracker:
    """Track conversion progress"""
    def __init__(self, job_id):
        self.job_id = job_id
        self.stage = "initializing"
        self.progress = 0
        self.message = "Starting conversion..."
        self.total_frames = 0
        self.processed_frames = 0
    
    def update(self, stage, progress, message, total_frames=0, processed_frames=0):
        """Update progress status"""
        self.stage = stage
        self.progress = min(100, max(0, progress))
        self.message = message
        self.total_frames = total_frames
        self.processed_frames = processed_frames
        
        conversion_status[self.job_id] = {
            'stage': self.stage,
            'progress': self.progress,
            'message': self.message,
            'total_frames': self.total_frames,
            'processed_frames': self.processed_frames,
            'status': 'processing'
        }


class ProgressVideoConverter(CompactVideoToLottie):
    """Extended converter with progress tracking using compact logic"""
    
    def __init__(self, video_path, output_path, fps, quality, max_width, tracker):
        super().__init__(video_path, output_path, fps, quality, max_width)
        self.tracker = tracker
    
    def extract_frames(self):
        """Override extract_frames with progress tracking and better error handling"""
        self.tracker.update("extracting", 10, "Opening video file...")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video file. The video may be corrupted or in an unsupported format."
            )

        # Test if we can actually read frames
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            cap.release()
            raise RuntimeError(
                "Cannot decode video frames. The video codec may not be supported by OpenCV."
            )
        
        # Reset to beginning after test
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(orig_fps / self.fps))

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if w > self.max_width:
            scale = self.max_width / w
            w = self.max_width
            h = int(h * scale)

        self.tracker.update("extracting", 15, 
                          f"Video: {w}x{h}, {orig_fps:.1f}fps, {total_frames} frames")
        
        frames = []
        idx = 0
        expected_frames = total_frames // interval

        self.tracker.update("extracting", 20, 
                          f"Extracting frames (every {interval} frame)...",
                          expected_frames, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % interval == 0:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
                # Update progress (20-50% range for extraction)
                progress = 20 + int((len(frames) / max(1, expected_frames)) * 30)
                self.tracker.update("extracting", progress, 
                                  f"Extracted {len(frames)} frames...",
                                  expected_frames, len(frames))

            idx += 1

        cap.release()
        
        if len(frames) == 0:
            raise RuntimeError("No frames were extracted from the video")
        
        self.tracker.update("extracting", 50, 
                          f"Extraction complete: {len(frames)} frames",
                          len(frames), len(frames))
        
        return frames, w, h

    def smooth_frames(self, frames, alpha=0.7):
        """Override with progress tracking"""
        self.tracker.update("smoothing", 52, "Smoothing frames for better quality...")
        smoothed = [frames[0]]
        total = len(frames)
        
        for i in range(1, total):
            prev = smoothed[-1].astype(np.float32)
            curr = frames[i].astype(np.float32)
            blended = cv2.addWeighted(curr, alpha, prev, 1 - alpha, 0)
            smoothed.append(blended.astype(np.uint8))
            
            # Update progress (52-58% range)
            progress = 52 + int((i / total) * 6)
            if i % 5 == 0:
                self.tracker.update("smoothing", progress, 
                                  f"Smoothing frame {i}/{total}...")
        
        return smoothed

    def deduplicate(self, frames):
        """Override with progress tracking"""
        self.tracker.update("analyzing", 60, "Analyzing frames for optimization...")
        assets = []
        frame_refs = []
        cache = []
        total = len(frames)

        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            motion = np.mean(edges)

            matched = False
            for j, (g, _) in enumerate(cache):
                mse = np.mean((gray.astype(float) - g.astype(float)) ** 2)
                similarity = 1.0 - (np.sqrt(mse) / 255.0)

                if similarity > 0.999 and motion < 2:
                    frame_refs.append(j)
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

            # Update progress (60-85% range)
            progress = 60 + int((i / total) * 25)
            if i % 5 == 0:
                self.tracker.update("analyzing", progress, 
                                  f"Analyzing frame {i+1}/{total}...",
                                  total, i+1)

        savings = ((1 - len(assets)/total) * 100)
        self.tracker.update("analyzing", 85, 
                          f"Found {len(assets)} unique frames (saved {savings:.1f}%)")
        
        return assets, frame_refs

    def convert(self):
        """Override convert with progress tracking"""
        self.tracker.update("starting", 5, "Initializing compact converter...")
        
        # Extract frames
        frames, w, h = self.extract_frames()
        
        # Smooth frames
        frames = self.smooth_frames(frames)
        
        # Deduplicate and encode
        assets, refs = self.deduplicate(frames)
        
        self.tracker.update("finalizing", 90, "Creating Lottie structure...")
        
        # Build Lottie
        lottie_json = self.build_lottie(assets, refs, w, h)
        
        self.tracker.update("saving", 95, "Saving compact Lottie JSON file...")
        
        with open(self.output_path, 'w') as f:
            json.dump(lottie_json, f, separators=(',', ':'))
        
        file_size = os.path.getsize(self.output_path)
        file_size_mb = file_size / (1024 * 1024)
        
        self.tracker.update("complete", 100, 
                          f"Complete! {len(frames)} frames, {file_size_mb:.2f}MB")
        
        return self.output_path


def convert_video_background(job_id, video_path, output_path, fps, quality, max_width):
    """Background task for video conversion"""
    preprocessed_path = None
    try:
        tracker = ProgressTracker(job_id)
        
        # Preprocess video to ensure compatibility
        preprocessed_path = video_path + "_processed.mp4"
        
        try:
            preprocess_video(video_path, preprocessed_path, tracker)
        except Exception as e:
            raise RuntimeError(f"Video preprocessing failed: {str(e)}")
        
        # Use preprocessed video for conversion
        converter = ProgressVideoConverter(
            video_path=preprocessed_path,
            output_path=output_path,
            fps=fps,
            quality=quality,
            max_width=max_width,
            tracker=tracker
        )
        
        output_file = converter.convert()
        
        # Mark as complete
        conversion_status[job_id]['status'] = 'complete'
        conversion_status[job_id]['output_file'] = output_file
        conversion_status[job_id]['file_size'] = os.path.getsize(output_file)
        
    except Exception as e:
        conversion_status[job_id] = {
            'status': 'error',
            'error': str(e),
            'stage': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}'
        }
    finally:
        # Cleanup temporary files
        for path in [video_path, preprocessed_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/convert', methods=['POST'])
def convert_video():
    """Handle video upload and conversion"""
    try:
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Get conversion parameters
        fps = int(request.form.get('fps', 15))
        quality = int(request.form.get('quality', 85))
        max_width = int(request.form.get('max_width', 608))
        
        # Validate parameters
        if not (1 <= fps <= 60):
            return jsonify({'error': 'FPS must be between 1 and 60'}), 400
        
        if not (1 <= quality <= 100):
            return jsonify({'error': 'Quality must be between 1 and 100'}), 400
        
        if not (100 <= max_width <= 2000):
            return jsonify({'error': 'Max width must be between 100 and 2000'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        job_id = f"{int(time.time() * 1000)}_{filename}"
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"input_{job_id}")
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{job_id}.json")
        
        file.save(input_path)
        
        # Start conversion in background
        thread = threading.Thread(
            target=convert_video_background,
            args=(job_id, input_path, output_path, fps, quality, max_width)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'message': 'Conversion started'
        }), 202
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get conversion status"""
    if job_id not in conversion_status:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(conversion_status[job_id]), 200


@app.route('/api/download/<job_id>', methods=['GET'])
def download_file(job_id):
    """Download the converted Lottie file"""
    if job_id not in conversion_status:
        return jsonify({'error': 'Job not found'}), 404
    
    status = conversion_status[job_id]
    
    if status['status'] != 'complete':
        return jsonify({'error': 'Conversion not complete'}), 400
    
    output_file = status['output_file']
    
    if not os.path.exists(output_file):
        return jsonify({'error': 'Output file not found'}), 404
    
    return send_file(
        output_file,
        as_attachment=True,
        download_name=f'lottie_{job_id}.json',
        mimetype='application/json'
    )


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)