# Video to Lottie Converter

Convert your videos into Lottie animations with a professional web interface.

![Video to Lottie Converter](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.0-green.svg)

## 🚀 Features

- **Professional UI** - Modern, responsive design inspired by LottieFiles
- **Real-time Progress** - Live updates during conversion
- **Smart Optimization** - Duplicate frame detection for smaller file sizes
- **WebP Compression** - Optimized image encoding
- **Customizable Settings** - Adjust FPS, quality, and resolution
- **Drag & Drop** - Easy file upload
- **Free & Open Source** - No limits, no watermarks

## 📋 Requirements

- Python 3.11+
- OpenCV
- Flask
- Pillow
- NumPy

## 🛠️ Local Setup

1. **Clone or download this repository**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python app.py
```

4. **Open your browser:**
Navigate to `http://localhost:5000`

## 🌐 Deploy to Render (Free Tier)

1. **Push your code to GitHub**

2. **Create a new Web Service on Render:**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure the service:**
   - **Name:** `video-to-lottie` (or your preferred name)
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --timeout 300 --workers 2`
   - **Instance Type:** `Free`

4. **Advanced Settings (Optional):**
   - Add environment variable: `PYTHON_VERSION` = `3.11.7`
   - Increase timeout if needed for large files

5. **Click "Create Web Service"**

Your app will be live at: `https://your-service-name.onrender.com`

## 📖 Usage

1. **Upload Video:**
   - Drag and drop or click to select a video file
   - Supported formats: MP4, MOV, AVI, WebM, MKV
   - Max file size: 100MB

2. **Configure Settings:**
   - **FPS:** Frame rate (5-60, default: 15)
   - **Quality:** WebP quality (50-100, default: 85)
   - **Max Width:** Maximum width in pixels (320-1920, default: 608)

3. **Convert:**
   - Click "Start Conversion"
   - Watch real-time progress
   - Download your Lottie JSON when complete

4. **Use Your Lottie:**
   - Upload to [LottieFiles](https://lottiefiles.com)
   - Use in web projects with [lottie-web](https://github.com/airbnb/lottie-web)
   - Use in mobile apps with [lottie-ios](https://github.com/airbnb/lottie-ios) or [lottie-android](https://github.com/airbnb/lottie-android)

## 🎨 Customization

### Modify conversion settings in `videotolottie.py`:
```python
converter = VideoToLottieConverter(
    video_path=video_path,
    output_path=output_path,
    fps=15,              # Adjust default FPS
    quality=85,          # Adjust default quality
    max_width=608        # Adjust default max width
)
```

### Customize UI in `static/style.css`:
- Change color scheme via CSS variables
- Modify animations and transitions
- Adjust responsive breakpoints

## 🔧 API Endpoints

### `POST /api/convert`
Upload and convert a video file.

**Parameters:**
- `video` (file): Video file to convert
- `fps` (int): Target frame rate (1-60)
- `quality` (int): WebP quality (1-100)
- `max_width` (int): Maximum width (100-2000)

**Response:**
```json
{
  "job_id": "1234567890_video.mp4",
  "message": "Conversion started"
}
```

### `GET /api/status/<job_id>`
Get conversion progress.

**Response:**
```json
{
  "stage": "encoding",
  "progress": 75,
  "message": "Encoding frame 45/60...",
  "status": "processing"
}
```

### `GET /api/download/<job_id>`
Download the converted Lottie JSON file.

## 📊 File Size Optimization

The converter uses several techniques to minimize file size:

1. **Frame Sampling** - Reduces frames based on target FPS
2. **Resolution Scaling** - Limits maximum width
3. **Duplicate Detection** - Reuses identical frames
4. **WebP Compression** - Efficient image encoding
5. **JSON Minification** - Removes unnecessary whitespace

## ⚙️ Render Configuration

### Free Tier Limitations:
- **Memory:** 512 MB RAM
- **CPU:** Shared CPU
- **Storage:** Ephemeral (temporary files only)
- **Sleep:** Inactive services sleep after 15 minutes

### Optimization Tips:
1. Set reasonable file size limits (100MB default)
2. Use temporary storage (files are auto-deleted)
3. Implement timeout handling for large files
4. Consider upgrading for production use

## 🐛 Troubleshooting

**Service sleeps on free tier:**
- First request after sleep may be slow (cold start)
- Consider using a uptime monitor like [UptimeRobot](https://uptimerobot.com/)

**Memory errors:**
- Reduce max file size
- Lower default resolution
- Decrease FPS

**Timeout errors:**
- Increase timeout in Procfile
- Process smaller videos
- Upgrade to paid tier for longer processing

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 💡 Credits

Built with:
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [OpenCV](https://opencv.org/) - Video processing
- [Pillow](https://python-pillow.org/) - Image processing
- [Lottie](https://airbnb.design/lottie/) - Animation format

---

Made with ❤️ for the animation community
