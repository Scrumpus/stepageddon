#!/bin/bash

# Beat Sync - Setup Script
# This script helps you set up the project quickly

set -e

echo "🎮 Beat Sync Setup"
echo "=================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓${NC} .env file created"
    echo ""
    echo -e "${YELLOW}Please edit .env and add your API keys:${NC}"
    echo "  - ANTHROPIC_API_KEY (required)"
    echo "  - SPOTIFY_CLIENT_ID (optional)"
    echo "  - SPOTIFY_CLIENT_SECRET (optional)"
    echo ""
    read -p "Press enter when you've added your API keys..."
fi

# Check if backend/.env exists
if [ ! -f backend/.env ]; then
    echo -e "${YELLOW}Creating backend/.env file...${NC}"
    cp backend/.env.example backend/.env
    
    # Copy API keys from root .env if they exist
    if [ -f .env ]; then
        source .env
        if [ ! -z "$ANTHROPIC_API_KEY" ]; then
            sed -i "s/ANTHROPIC_API_KEY=.*/ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY/" backend/.env
        fi
        if [ ! -z "$SPOTIFY_CLIENT_ID" ]; then
            sed -i "s/SPOTIFY_CLIENT_ID=.*/SPOTIFY_CLIENT_ID=$SPOTIFY_CLIENT_ID/" backend/.env
        fi
        if [ ! -z "$SPOTIFY_CLIENT_SECRET" ]; then
            sed -i "s/SPOTIFY_CLIENT_SECRET=.*/SPOTIFY_CLIENT_SECRET=$SPOTIFY_CLIENT_SECRET/" backend/.env
        fi
    fi
    echo -e "${GREEN}✓${NC} backend/.env file created"
fi

echo ""
echo "Choose setup method:"
echo "1) Docker (Recommended)"
echo "2) Local Development"
read -p "Enter choice [1-2]: " choice

case $choice in
    1)
        echo ""
        echo -e "${YELLOW}Setting up with Docker...${NC}"
        
        # Check if Docker is installed
        if ! command -v docker &> /dev/null; then
            echo -e "${RED}✗${NC} Docker is not installed"
            echo "Please install Docker: https://docs.docker.com/get-docker/"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            echo -e "${RED}✗${NC} Docker Compose is not installed"
            echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
            exit 1
        fi
        
        echo -e "${GREEN}✓${NC} Docker found"
        
        # Build and start containers
        echo ""
        echo "Building Docker containers..."
        docker-compose build
        
        echo ""
        echo "Starting services..."
        docker-compose up -d
        
        echo ""
        echo -e "${GREEN}✓ Setup complete!${NC}"
        echo ""
        echo "Services running:"
        echo "  Frontend: http://localhost"
        echo "  Backend:  http://localhost:8000"
        echo ""
        echo "To view logs: docker-compose logs -f"
        echo "To stop:      docker-compose down"
        ;;
        
    2)
        echo ""
        echo -e "${YELLOW}Setting up for Local Development...${NC}"
        
        # Check Python
        if ! command -v python3 &> /dev/null; then
            echo -e "${RED}✗${NC} Python 3 is not installed"
            exit 1
        fi
        echo -e "${GREEN}✓${NC} Python 3 found"
        
        # Check Node.js
        if ! command -v node &> /dev/null; then
            echo -e "${RED}✗${NC} Node.js is not installed"
            exit 1
        fi
        echo -e "${GREEN}✓${NC} Node.js found"
        
        # Setup backend
        echo ""
        echo "Setting up backend..."
        cd backend
        
        if [ ! -d "venv" ]; then
            python3 -m venv venv
        fi
        
        source venv/bin/activate
        pip install -r requirements.txt
        
        echo -e "${GREEN}✓${NC} Backend setup complete"
        cd ..
        
        # Setup frontend
        echo ""
        echo "Setting up frontend..."
        cd frontend
        npm install
        echo -e "${GREEN}✓${NC} Frontend setup complete"
        cd ..
        
        echo ""
        echo -e "${GREEN}✓ Setup complete!${NC}"
        echo ""
        echo "To start development servers:"
        echo ""
        echo "Terminal 1 (Backend):"
        echo "  cd backend"
        echo "  source venv/bin/activate"
        echo "  python main.py"
        echo ""
        echo "Terminal 2 (Frontend):"
        echo "  cd frontend"
        echo "  npm run dev"
        echo ""
        echo "Then open: http://localhost:3000"
        ;;
        
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎮 Happy gaming!"
