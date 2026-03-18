#!/bin/bash

echo "🚀 FaceAI Deploy Started..."

cd /home/ec2-user/face-scanner-app || exit

echo "📥 Pulling latest code..."
git pull origin main

echo "🔁 Restarting service..."
sudo systemctl restart faceai

echo "📊 Service status:"
sudo systemctl status faceai --no-pager

echo "✅ Deploy Completed!"