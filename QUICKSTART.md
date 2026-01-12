# 🚀 Quick Start Guide

## ⚡ 60 Seconds to Deploy

### Option 1: Deploy to Render (Recommended)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Video to Lottie Converter"
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Deploy on Render:**
   - Visit https://dashboard.render.com/
   - Click "New +" → "Web Service"
   - Connect your GitHub repo
   - Instance Type: **Free**
   - Click "Create Web Service"
   - ✅ Done! Your app will be live in ~5 minutes

### Option 2: Run Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server:**
   ```bash
   python app.py
   ```

3. **Open browser:**
   ```
   http://localhost:5000
   ```

## 📖 How to Use

1. **Upload a Video**
   - Drag & drop or click to browse
   - Supports: MP4, MOV, AVI, WebM, MKV
   - Max size: 100MB

2. **Adjust Settings** (Optional)
   - Frame Rate: 5-60 FPS
   - Quality: 50-100
   - Max Width: 320-1920px

3. **Convert**
   - Click "Start Conversion"
   - Watch real-time progress
   - Download your Lottie JSON

4. **Use Your Lottie**
   - Upload to LottieFiles.com
   - Use in websites/apps
   - Share with your team

## 🎯 File Structure

```
video-to-lottie/
├── app.py              # Flask backend with API
├── videotolottie.py    # Video converter engine
├── index.html          # Main web page
├── static/
│   ├── style.css       # Modern UI styling
│   └── script.js       # Client-side logic
├── requirements.txt    # Python dependencies
├── Procfile           # Render deployment config
├── runtime.txt        # Python version
└── README.md          # Full documentation
```

## 🌐 Your Live URL

After deploying to Render:
```
https://YOUR-SERVICE-NAME.onrender.com
```

## 💡 Pro Tips

- **Free Tier:** Service sleeps after 15 min. First request may be slow (~30s)
- **File Size:** Keep videos under 50MB for faster processing
- **Settings:** Lower FPS = smaller files, faster conversion
- **Quality:** 85 is optimal balance of quality/size

## 🆘 Troubleshooting

**App won't start locally?**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**Deployment failed?**
- Check Python version in runtime.txt matches Render
- Verify all files are committed to git
- Check Render build logs for errors

**Conversion timeout?**
- Use smaller video files
- Lower FPS/quality settings
- Consider upgrading Render instance

## 📞 Need Help?

Check the full README.md for:
- Detailed API documentation
- Advanced configuration
- Optimization tips
- Feature customization

---

**Ready to convert? Let's go! 🎬✨**
