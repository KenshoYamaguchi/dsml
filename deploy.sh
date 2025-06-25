#!/bin/bash

# Azure deployment script
echo "Starting deployment..."

# Install Python dependencies
echo "Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p uploads
mkdir -p model

# Set environment variables for matplotlib
export MPLBACKEND=Agg

echo "Deployment completed!"