#!/bin/bash
# Deploy lightweight face recognition model with automatic k3s deployment

echo "🚀 Deploying lightweight face recognition model..."

# Build lightweight Docker image
sudo docker build -t face-recognition-lightweight:v1 .

# Save as tar (should be much smaller)
sudo docker save face-recognition-lightweight:v1 -o face-recognition-lightweight-v1.tar

# Fix permissions
sudo chown $USER:$USER face-recognition-lightweight-v1.tar

# Check size
echo "📊 Image size:"
ls -lh face-recognition-lightweight-v1.tar

echo "✅ Lightweight deployment ready!"
echo "Image size should be under 1GB"

# Ask if user wants to auto-deploy to k3s
echo ""
read -p "🤖 Auto-deploy to k3s now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting automatic k3s deployment..."
    python3 auto_deploy.py --model-version 1
else
    echo "📋 To deploy manually later, run:"
    echo "   python3 auto_deploy.py --model-version 1"
fi
