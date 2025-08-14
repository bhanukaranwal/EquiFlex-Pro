#!/bin/bash

# EquiFlex Pro Setup Script
set -e

echo "🚀 Setting up EquiFlex Pro Trading Bot..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | awk -F. '{print $1"."$2}')
required_version="3.9"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "❌ Python 3.9+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "🔧 Installing development dependencies..."
pip install -e ".[dev]"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs models data/cache configs/local web/uploads

# Copy example configurations
echo "⚙️  Setting up configurations..."
cp configs/config.yaml configs/local/config.yaml
cp .env.example .env

# Initialize database (if Docker is available)
if command -v docker &> /dev/null; then
    echo "🐳 Setting up database with Docker..."
    docker-compose up -d postgres redis
    
    # Wait for database to be ready
    echo "⏳ Waiting for database to be ready..."
    sleep 10
    
    # Run database migrations
    python scripts/init_database.py
else
    echo "⚠️  Docker not found. Please set up PostgreSQL manually."
fi

# Set up pre-commit hooks
echo "🔍 Setting up pre-commit hooks..."
pre-commit install

# Download sample data (optional)
echo "📊 Downloading sample data..."
python scripts/download_sample_data.py

# Run initial tests
echo "🧪 Running initial tests..."
pytest tests/test_basic.py -v

echo ""
echo "🎉 EquiFlex Pro setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Configure your API keys in .env file"
echo "2. Review and update configs/local/config.yaml"
echo "3. Start the trading engine: python -m src.core.engine"
echo "4. Start the API server: python -m src.api.app"
echo "5. Open http://localhost:8000 in your browser"
echo ""
echo "📖 Documentation: docs/README.md"
echo "🐛 Report issues: https://github.com/bhanukaranwal/EquiFlexPro/issues"