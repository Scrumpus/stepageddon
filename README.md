# Stepaggedon

DDR-style browser game that let's you upload any song (url or file) and generates step chart. Built with React and FastAPI

For production deployment instructions see [DEPLOY.md](./DEPLOY.md).

### Local Development

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Run server
python main.py
```

Backend will run at http://localhost:8000

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will run at http://localhost:3000

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm run test
```