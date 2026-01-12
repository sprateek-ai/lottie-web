#!/usr/bin/env python3
"""
Flask Web Application for Video to Lottie Converter
Provides RESTful API with real-time progress updates
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
from videotolottie import VideoToLottieConverter
import cv2
import base64
from io import BytesIO
from PIL import Image
import numpy as np

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


class ProgressVideoConverter(VideoToLottieConverter):
    """Extended converter with progress tracking"""
    
    def __init__(self, video_path, output_path, fps, quality, max_width, tracker):
        super().__init__(video_path, output_path, fps, quality, max_width)
        self.tracker = tracker
    
    def extract_frames(self):
        """Override extract_frames with progress tracking"""
        self.tracker.update("extracting", 10, "Opening video file...")
        
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.tracker.update("extracting", 15, 
                          f"Video: {width}x{height}, {original_fps:.1f}fps, {total_frames} frames")
        
        # Calculate frame sampling rate
        frame_interval = max(1, int(original_fps / self.target_fps))
        
        # Calculate output dimensions
        if width <= 0 or height <= 0:
            raise ValueError("Could not read video dimensions. Please try a different video format.")
            
        if width > self.max_width:
            output_width = self.max_width
            output_height = int(height * (self.max_width / width))
        else:
            output_width = width
            output_height = height
        
        frames = []
        frame_count = 0
        extracted_count = 0
        expected_frames = total_frames // frame_interval
        
        self.tracker.update("extracting", 20, 
                          f"Extracting frames (every {frame_interval} frame)...",
                          expected_frames, 0)
        
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
                
                # Update progress (20-50% range for extraction)
                progress = 20 + int((extracted_count / max(1, expected_frames)) * 30)
                self.tracker.update("extracting", progress, 
                                  f"Extracted {extracted_count} frames...",
                                  expected_frames, extracted_count)
            
            frame_count += 1
        
        cap.release()
        
        self.tracker.update("extracting", 50, 
                          f"Extraction complete: {extracted_count} frames",
                          extracted_count, extracted_count)
        
        return frames, output_width, output_height
    
    def detect_duplicate_frames(self, frames, threshold=0.98):
        """Override with progress tracking"""
        self.tracker.update("analyzing", 55, "Analyzing frames for optimization...")
        
        frame_map = {}
        unique_frames = []
        unique_ids = []
        
        total_frames = len(frames)
        
        for idx, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            is_duplicate = False
            for unique_idx, (unique_gray, unique_frame) in enumerate(unique_frames):
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
            
            # Update progress (55-65% range)
            progress = 55 + int((idx / total_frames) * 10)
            self.tracker.update("analyzing", progress, 
                              f"Analyzing frame {idx+1}/{total_frames}...",
                              total_frames, idx+1)
        
        if total_frames == 0:
            self.tracker.update("analyzing", 65, "No frames to analyze")
            return {}, []
            
        savings = ((1 - len(unique_frames)/total_frames) * 100)
        self.tracker.update("analyzing", 65, 
                          f"Found {len(unique_frames)} unique frames (saved {savings:.1f}%)")
        
        return frame_map, [f for _, f in unique_frames]
    
    def create_lottie_json(self, frames, width, height):
        """Override with progress tracking"""
        num_frames = len(frames)
        
        # Detect duplicates
        frame_map, unique_frames = self.detect_duplicate_frames(frames)
        
        self.tracker.update("encoding", 70, "Converting frames to WebP format...")
        
        # Create assets
        assets = []
        for idx, frame in enumerate(unique_frames):
            base64_image = self.frame_to_base64_webp(frame)
            
            asset = {
                "id": f"image_{idx}",
                "w": width,
                "h": height,
                "p": base64_image,
                "e": 1
            }
            assets.append(asset)
            
            # Update progress (70-90% range)
            progress = 70 + int((idx / len(unique_frames)) * 20)
            self.tracker.update("encoding", progress, 
                              f"Encoding frame {idx+1}/{len(unique_frames)}...",
                              len(unique_frames), idx+1)
        
        self.tracker.update("finalizing", 90, "Creating Lottie structure...")
        
        # Create layers
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
        
        # Create main structure
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
        
        self.tracker.update("finalizing", 95, "Lottie structure created")
        
        return lottie_json
    
    def convert(self):
        """Override convert with progress tracking"""
        self.tracker.update("starting", 5, "Initializing converter...")
        
        # Extract frames
        frames, width, height = self.extract_frames()
        
        # Create Lottie JSON
        lottie_data = self.create_lottie_json(frames, width, height)
        
        # Save to file
        self.tracker.update("saving", 98, "Saving Lottie JSON file...")
        
        with open(self.output_path, 'w') as f:
            json.dump(lottie_data, f, separators=(',', ':'))
        
        file_size = os.path.getsize(self.output_path)
        file_size_mb = file_size / (1024 * 1024)
        
        self.tracker.update("complete", 100, 
                          f"Complete! {len(frames)} frames, {file_size_mb:.2f}MB")
        
        return self.output_path


def convert_video_background(job_id, video_path, output_path, fps, quality, max_width):
    """Background task for video conversion"""
    try:
        tracker = ProgressTracker(job_id)
        
        converter = ProgressVideoConverter(
            video_path=video_path,
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
