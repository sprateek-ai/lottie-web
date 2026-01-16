// State Management
let selectedFile = null;
let currentJobId = null;
let pollingInterval = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const videoInput = document.getElementById('videoInput');
const settingsPanel = document.getElementById('settingsPanel');
const progressPanel = document.getElementById('progressPanel');
const completePanel = document.getElementById('completePanel');
const errorPanel = document.getElementById('errorPanel');
const fileInfo = document.getElementById('fileInfo');
const fpsInput = document.getElementById('fpsInput');
const fpsValue = document.getElementById('fpsValue');
const qualityInput = document.getElementById('qualityInput');
const qualityValue = document.getElementById('qualityValue');
const widthInput = document.getElementById('widthInput');
const widthValue = document.getElementById('widthValue');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    setupDragAndDrop();
    setupSliders();
});

// Event Listeners
function setupEventListeners() {
    videoInput.addEventListener('change', handleFileSelect);
}

function setupDragAndDrop() {
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    uploadArea.addEventListener('click', () => {
        videoInput.click();
    });
}

function setupSliders() {
    fpsInput.addEventListener('input', (e) => {
        fpsValue.textContent = e.target.value;
    });

    qualityInput.addEventListener('input', (e) => {
        qualityValue.textContent = e.target.value;
    });

    widthInput.addEventListener('input', (e) => {
        widthValue.textContent = e.target.value;
    });
}

// File Handling
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    // Validate file type
    const allowedTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm', 'video/x-matroska'];
    if (!allowedTypes.includes(file.type)) {
        showError('Invalid file type. Please select a video file (MP4, MOV, AVI, WebM, or MKV).');
        return;
    }

    // Validate file size (100MB max)
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File size exceeds 100MB limit. Please select a smaller file.');
        return;
    }

    selectedFile = file;
    showSettings(file);
}

function showSettings(file) {
    // Format file size
    const sizeInMB = (file.size / (1024 * 1024)).toFixed(2);

    // Display file info
    fileInfo.innerHTML = `
        <div style="display: flex; align-items: center; gap: 1rem;">
            <svg style="width: 40px; height: 40px; color: var(--primary);" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M15 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V7L15 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M14 2V8H20" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M10 15L12 17L16 13" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <div style="flex: 1;">
                <div style="font-weight: 600; margin-bottom: 0.25rem;">${file.name}</div>
                <div style="font-size: 0.875rem; color: var(--text-tertiary);">${sizeInMB} MB</div>
            </div>
        </div>
    `;

    // Show settings panel
    uploadArea.style.display = 'none';
    settingsPanel.style.display = 'block';
}

// Conversion Process
async function startConversion() {
    if (!selectedFile) {
        showError('No file selected');
        return;
    }

    // Get settings
    const fps = parseInt(fpsInput.value);
    const quality = parseInt(qualityInput.value);
    const maxWidth = parseInt(widthInput.value);

    // Prepare form data
    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('fps', fps);
    formData.append('quality', quality);
    formData.append('max_width', maxWidth);

    // Show progress panel
    settingsPanel.style.display = 'none';
    progressPanel.style.display = 'block';

    try {
        // Start conversion
        const response = await fetch('/api/convert', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Conversion failed');
        }

        const data = await response.json();
        currentJobId = data.job_id;

        // Start polling for status
        startPolling();

    } catch (error) {
        showError(error.message);
    }
}

function startPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }

    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/status/${currentJobId}`);

            if (!response.ok) {
                throw new Error('Failed to get status');
            }

            const status = await response.json();
            updateProgress(status);

            if (status.status === 'complete') {
                clearInterval(pollingInterval);
                showComplete(status);
            } else if (status.status === 'error') {
                clearInterval(pollingInterval);
                showError(status.error || 'Conversion failed');
            }

        } catch (error) {
            clearInterval(pollingInterval);
            showError(error.message);
        }
    }, 500); // Poll every 500ms
}

function updateProgress(status) {
    const { stage, progress, message, total_frames, processed_frames } = status;

    // Update stage with preprocessing support
    const stageName = {
        'initializing': 'Initializing',
        'preprocessing': 'Converting Video Format',  // ← Added preprocessing stage
        'starting': 'Starting',
        'extracting': 'Extracting Frames',
        'smoothing': 'Smoothing Frames',  // ← Added smoothing stage
        'analyzing': 'Analyzing & Optimizing',
        'encoding': 'Encoding',
        'finalizing': 'Finalizing',
        'saving': 'Saving',
        'complete': 'Complete'
    }[stage] || 'Processing';

    document.getElementById('progressStage').textContent = stageName;
    document.getElementById('progressPercentage').textContent = `${Math.round(progress)}%`;
    document.getElementById('progressMessage').textContent = message;
    document.getElementById('progressBar').style.width = `${progress}%`;

    // Update details if available
    if (total_frames > 0) {
        document.getElementById('progressDetails').innerHTML = `
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>Progress:</span>
                <span>${processed_frames} / ${total_frames} frames</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Stage:</span>
                <span>${stageName}</span>
            </div>
        `;
    } else {
        // Show message for preprocessing stage
        document.getElementById('progressDetails').innerHTML = `
            <div style="text-align: center; padding: 1rem;">
                <span>${message}</span>
            </div>
        `;
    }
}

function showComplete(status) {
    progressPanel.style.display = 'none';
    completePanel.style.display = 'block';

    // Display stats
    const fileSizeMB = (status.file_size / (1024 * 1024)).toFixed(2);

    document.getElementById('completeStats').innerHTML = `
        <div class="complete-stat">
            <div class="complete-stat-label">File Size</div>
            <div class="complete-stat-value">${fileSizeMB} MB</div>
        </div>
        <div class="complete-stat">
            <div class="complete-stat-label">Total Frames</div>
            <div class="complete-stat-value">${status.total_frames || 'N/A'}</div>
        </div>
        <div class="complete-stat">
            <div class="complete-stat-label">Status</div>
            <div class="complete-stat-value">✓ Ready</div>
        </div>
    `;

    // Setup download button
    const downloadBtn = document.getElementById('downloadBtn');
    downloadBtn.onclick = () => downloadFile();
}

function downloadFile() {
    const downloadUrl = `/api/download/${currentJobId}`;
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `lottie_${currentJobId}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function showError(message) {
    // Hide all panels
    uploadArea.style.display = 'none';
    settingsPanel.style.display = 'none';
    progressPanel.style.display = 'none';
    completePanel.style.display = 'none';

    // Show error panel
    errorPanel.style.display = 'block';
    document.getElementById('errorMessage').textContent = message;
}

function resetConverter() {
    // Clear state
    selectedFile = null;
    currentJobId = null;
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }

    // Reset input
    videoInput.value = '';

    // Reset sliders
    fpsInput.value = 15;
    fpsValue.textContent = '15';
    qualityInput.value = 85;
    qualityValue.textContent = '85';
    widthInput.value = 608;
    widthValue.textContent = '608';

    // Hide all panels
    settingsPanel.style.display = 'none';
    progressPanel.style.display = 'none';
    completePanel.style.display = 'none';
    errorPanel.style.display = 'none';

    // Show upload area
    uploadArea.style.display = 'block';
}

// Smooth scroll for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});