# Contributing to Beat Sync

Thank you for your interest in contributing to Beat Sync! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards others

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title**: Describe the issue concisely
- **Steps to reproduce**: Detailed steps to reproduce the behavior
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Screenshots**: If applicable
- **Environment**: OS, browser, versions
- **Logs**: Relevant error messages or logs

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Clear description**: What feature you'd like to see
- **Use case**: Why this feature would be useful
- **Mockups**: Visual designs if applicable
- **Implementation ideas**: Technical approach (optional)

### Pull Requests

1. **Fork the repository**
2. **Create a branch**: `git checkout -b feature/AmazingFeature`
3. **Make your changes**
4. **Test thoroughly**
5. **Commit**: `git commit -m 'Add some AmazingFeature'`
6. **Push**: `git push origin feature/AmazingFeature`
7. **Open a Pull Request**

## Development Setup

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker (optional)
- Git

### Local Setup

```bash
# Clone repository
git clone <your-fork>
cd beat-sync

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
python main.py

# Frontend setup (in new terminal)
cd frontend
npm install
npm run dev
```

## Coding Standards

### Python (Backend)

- Follow PEP 8 style guide
- Use type hints where applicable
- Write docstrings for functions/classes
- Keep functions focused and small
- Use async/await for I/O operations

Example:
```python
async def analyze_audio(audio_path: str) -> Dict:
    """
    Analyze audio file and extract features.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary containing analysis results
    """
    # Implementation
```

### JavaScript/React (Frontend)

- Use functional components with hooks
- Keep components focused and reusable
- Use meaningful variable names
- Add comments for complex logic
- Follow existing project structure

Example:
```javascript
/**
 * Evaluates hit timing and returns judgment
 * @param {number} timingMs - Timing difference in milliseconds
 * @returns {string} Judgment: PERFECT, GOOD, OK, or MISS
 */
export const evaluateHit = (timingMs) => {
  // Implementation
};
```

### Git Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests

Examples:
```
Add Spotify preview support

- Implement preview URL extraction
- Add warning for preview-only tracks
- Update error messages

Fixes #123
```

## Testing

### Backend Tests

```bash
cd backend
pytest tests/ -v
```

### Frontend Tests

```bash
cd frontend
npm run test
```

### Manual Testing Checklist

- [ ] File upload works (MP3, WAV, OGG, FLAC)
- [ ] YouTube URL works
- [ ] Spotify URL works
- [ ] All difficulty levels work
- [ ] Gameplay is smooth (60 FPS)
- [ ] Hit detection is accurate
- [ ] Score calculation is correct
- [ ] Combo system works
- [ ] Results screen shows correct data
- [ ] Pause/resume works
- [ ] Return to menu works

## Project Structure

```
beat-sync/
├── backend/
│   ├── main.py              # Entry point
│   ├── core/                # Configuration
│   ├── services/            # Business logic
│   ├── routers/             # API endpoints
│   └── tests/               # Backend tests
│
├── frontend/
│   ├── src/
│   │   ├── screens/         # Game screens
│   │   ├── components/      # Reusable components
│   │   ├── utils/           # Utilities
│   │   └── config/          # Configuration
│   └── tests/               # Frontend tests
│
└── docs/                    # Documentation
```

## Areas for Contribution

### High Priority
- [ ] Unit tests (backend and frontend)
- [ ] Integration tests
- [ ] Performance optimizations
- [ ] Error handling improvements
- [ ] Documentation

### Features
- [ ] User accounts and authentication
- [ ] Chart editor
- [ ] Leaderboards
- [ ] Custom themes/skins
- [ ] Mobile support
- [ ] Multiplayer

### Bug Fixes
- Check GitHub issues for open bugs

### Documentation
- [ ] API documentation
- [ ] User guide
- [ ] Video tutorials
- [ ] Deployment guides

## Review Process

1. **Automated checks**: Linting, tests must pass
2. **Code review**: At least one maintainer review
3. **Testing**: Manual testing of changes
4. **Documentation**: Update docs if needed
5. **Merge**: Squash and merge to main

## Questions?

- Open an issue for questions
- Join our Discord (link in README)
- Email: support@beatsync.example.com

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in the app (if significant contribution)

Thank you for contributing to Beat Sync! 🎮🎵
