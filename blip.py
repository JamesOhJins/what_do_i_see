import os
from flask import Flask, request, render_template_string, jsonify
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import base64
import io

app = Flask(__name__)

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

@app.route('/', methods=['GET'])
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Image Analyzer with TTS</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center p-4">
    <div class="bg-white p-8 rounded-lg shadow-md w-full max-w-md">
        <h1 class="text-3xl font-bold mb-6 text-center text-gray-800">AI Image Analyzer</h1>
        
        <div id="image-container" class="mb-6 hidden">
            <img id="preview-image" class="w-full h-64 object-cover rounded-lg mb-2">
            <div class="flex justify-between">
                <button id="reupload-button" class="text-blue-500 hover:text-blue-700">
                    <i class="fas fa-redo-alt mr-2"></i>Reupload
                </button>
                <button id="remove-image" class="text-red-500 hover:text-red-700">
                    <i class="fas fa-trash-alt mr-2"></i>Remove
                </button>
            </div>
        </div>

        <div id="upload-container" class="space-y-4">
            <div class="flex items-center justify-center w-full">
                <label for="file-upload" class="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition duration-300 ease-in-out">
                    <div class="flex flex-col items-center justify-center pt-5 pb-6">
                        <i class="fas fa-cloud-upload-alt text-4xl text-gray-400 mb-3"></i>
                        <p class="mb-2 text-sm text-gray-500"><span class="font-semibold">Click to upload</span> or drag and drop</p>
                        <p class="text-xs text-gray-500">PNG, JPG or GIF (MAX. 5MB)</p>
                    </div>
                    <input id="file-upload" type="file" name="file" class="hidden" accept="image/*" />
                </label>
            </div>
            <button id="camera-button" class="w-full bg-green-500 hover:bg-green-600 text-white font-bold py-3 px-4 rounded-lg focus:outline-none focus:shadow-outline transition duration-300 ease-in-out">
                <i class="fas fa-camera mr-2"></i>Use Camera
            </button>
        </div>

        <video id="camera-feed" class="w-full rounded-lg mt-4 hidden"></video>
        <canvas id="canvas" class="hidden"></canvas>
        
        <div id="camera-controls" class="mt-4 space-x-2 hidden">
            <button id="capture-button" class="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg focus:outline-none focus:shadow-outline transition duration-300 ease-in-out">
                <i class="fas fa-camera mr-2"></i>Capture
            </button>
            <button id="switch-camera" class="bg-gray-500 hover:bg-gray-600 text-white font-bold py-2 px-4 rounded-lg focus:outline-none focus:shadow-outline transition duration-300 ease-in-out">
                <i class="fas fa-sync-alt mr-2"></i>Switch Camera
            </button>
        </div>

        <button id="analyze-button" class="w-full mt-6 bg-purple-500 hover:bg-purple-600 text-white font-bold py-3 px-4 rounded-lg focus:outline-none focus:shadow-outline transition duration-300 ease-in-out">
            <i class="fas fa-magic mr-2"></i>Analyze Image
        </button>

        <div id="loader" class="hidden mt-4">
            <div class="loader ease-linear rounded-full border-4 border-t-4 border-gray-200 h-12 w-12 mb-4 mx-auto"></div>
            <p class="text-center text-gray-600">Analyzing image...</p>
        </div>

        <div id="result" class="mt-6 p-4 bg-gray-100 rounded-lg hidden">
            <h2 class="text-xl font-semibold mb-2 text-gray-800">Analysis Result:</h2>
            <p id="result-text" class="text-gray-700"></p>
            <div class="mt-4 flex items-center">
                <label for="tts-toggle" class="mr-3 font-medium text-gray-700">Text-to-Speech:</label>
                <div class="relative inline-block w-10 mr-2 align-middle select-none transition duration-200 ease-in">
                    <input type="checkbox" name="tts-toggle" id="tts-toggle" class="toggle-checkbox absolute block w-6 h-6 rounded-full bg-white border-4 appearance-none cursor-pointer"/>
                    <label for="tts-toggle" class="toggle-label block overflow-hidden h-6 rounded-full bg-gray-300 cursor-pointer"></label>
                </div>
            </div>
            <button id="play-tts" class="mt-2 bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg focus:outline-none focus:shadow-outline transition duration-300 ease-in-out hidden">
                <i class="fas fa-play mr-2"></i>Play
            </button>
        </div>
    </div>

    <script>
        let imageData = null;
        let stream = null;
        let currentFacingMode = 'environment';
        let speechSynthesis = window.speechSynthesis;
        let speechUtterance = new SpeechSynthesisUtterance();

        const fileUpload = document.getElementById('file-upload');
        const imageContainer = document.getElementById('image-container');
        const previewImage = document.getElementById('preview-image');
        const uploadContainer = document.getElementById('upload-container');
        const cameraButton = document.getElementById('camera-button');
        const cameraFeed = document.getElementById('camera-feed');
        const cameraControls = document.getElementById('camera-controls');
        const canvas = document.getElementById('canvas');
        const captureButton = document.getElementById('capture-button');
        const switchCameraButton = document.getElementById('switch-camera');
        const analyzeButton = document.getElementById('analyze-button');
        const loader = document.getElementById('loader');
        const result = document.getElementById('result');
        const resultText = document.getElementById('result-text');
        const reuploadButton = document.getElementById('reupload-button');
        const removeImageButton = document.getElementById('remove-image');
        const ttsToggle = document.getElementById('tts-toggle');
        const playTtsButton = document.getElementById('play-tts');

        function showImage(src) {
            imageData = src;
            previewImage.src = src;
            imageContainer.classList.remove('hidden');
            uploadContainer.classList.add('hidden');
            cameraFeed.classList.add('hidden');
            cameraControls.classList.add('hidden');
        }

        function resetUI() {
            imageData = null;
            imageContainer.classList.add('hidden');
            uploadContainer.classList.remove('hidden');
            cameraFeed.classList.add('hidden');
            cameraControls.classList.add('hidden');
            result.classList.add('hidden');
            playTtsButton.classList.add('hidden');
        }

        fileUpload.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    showImage(e.target.result);
                }
                reader.readAsDataURL(file);
            }
        });

        async function setupCamera(facingMode) {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { facingMode: facingMode }
                });
                cameraFeed.srcObject = stream;
                cameraFeed.classList.remove('hidden');
                cameraControls.classList.remove('hidden');
                uploadContainer.classList.add('hidden');
                currentFacingMode = facingMode;
            } catch (err) {
                console.error('Error accessing camera:', err);
                alert('Unable to access camera. Please make sure you have given permission.');
            }
        }

        cameraButton.addEventListener('click', () => setupCamera(currentFacingMode));
        switchCameraButton.addEventListener('click', () => {
            currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
            setupCamera(currentFacingMode);
        });

        captureButton.addEventListener('click', () => {
            canvas.width = cameraFeed.videoWidth;
            canvas.height = cameraFeed.videoHeight;
            canvas.getContext('2d').drawImage(cameraFeed, 0, 0);
            showImage(canvas.toDataURL('image/jpeg'));
        });

        reuploadButton.addEventListener('click', resetUI);
        removeImageButton.addEventListener('click', resetUI);

        analyzeButton.addEventListener('click', () => {
            if (!imageData) {
                alert('Please upload or capture an image first.');
                return;
            }

            loader.classList.remove('hidden');
            result.classList.add('hidden');

            fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            })
            .then(response => response.json())
            .then(data => {
                loader.classList.add('hidden');
                result.classList.remove('hidden');
                resultText.textContent = data.description;
                if (ttsToggle.checked) {
                    playTtsButton.classList.remove('hidden');
                }
            })
            .catch(error => {
                loader.classList.add('hidden');
                result.classList.remove('hidden');
                resultText.textContent = 'An error occurred. Please try again.';
                console.error('Error:', error);
            });
        });

        ttsToggle.addEventListener('change', function() {
            if (this.checked && resultText.textContent) {
                playTtsButton.classList.remove('hidden');
            } else {
                playTtsButton.classList.add('hidden');
                speechSynthesis.cancel();
            }
        });

        playTtsButton.addEventListener('click', function() {
            if (speechSynthesis.speaking) {
                speechSynthesis.cancel();
                this.innerHTML = '<i class="fas fa-play mr-2"></i>Play';
            } else {
                speechUtterance.text = resultText.textContent;
                speechSynthesis.speak(speechUtterance);
                this.innerHTML = '<i class="fas fa-stop mr-2"></i>Stop';
            }
        });

        speechUtterance.onend = function() {
            playTtsButton.innerHTML = '<i class="fas fa-play mr-2"></i>Play';
        };

        // Drag and drop functionality
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            document.querySelector('label[for="file-upload"]').addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            document.querySelector('label[for="file-upload"]').addEventListener(eventName, unhighlight, false);
        });

        function highlight(e) {
            document.querySelector('label[for="file-upload"]').classList.add('bg-blue-100');
        }

        function unhighlight(e) {
            document.querySelector('label[for="file-upload"]').classList.remove('bg-blue-100');
        }

        document.querySelector('label[for="file-upload"]').addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            let dt = e.dataTransfer;
            let files = dt.files;
            handleFiles(files);
        }

        function handleFiles(files) {
            if (files[0]) {
                let reader = new FileReader();
                reader.readAsDataURL(files[0]);
                reader.onloadend = function() {
                    showImage(reader.result);
                }
            }
        }
    </script>
    <style>
        .toggle-checkbox:checked {
            @apply: right-0 border-green-400;
            right: 0;
            border-color: #68D391;
        }
        .toggle-checkbox:checked + .toggle-label {
            @apply: bg-green-400;
            background-color: #68D391;
        }
    </style>
</body>
</html>
    ''')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400
    
    # Decode base64 image
    image_data = data['image'].split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    inputs = blip_processor(image, return_tensors="pt")
    
    with torch.no_grad():
        generated_ids = blip_model.generate(**inputs, max_length=100)
        generated_caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    
    return jsonify({'description': generated_caption})

if __name__ == '__main__':
    app.run(debug=True)