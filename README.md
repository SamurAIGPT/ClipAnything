# Clip Anything

[![GitHub stars](https://img.shields.io/github/stars/SamurAIGPT/Clip-Anything?style=social)](https://github.com/SamurAIGPT/Clip-Anything/stargazers)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Multimodal AI video clipping** - Extract any moment from any video using natural language prompts. Just describe what you're looking for, and AI will find and clip it for you.

![Clip Anything Demo](https://github.com/user-attachments/assets/9689a74c-598a-4aab-b02e-54673941c2b9)

**Looking for an api to build your own Opus-clip style product ? Check the api below**

https://muapi.ai/playground/autocrop

https://muapi.ai/playground/ai-clipping

## Tutorials

- **YouTube**: [Watch Tutorial](https://youtu.be/R_3kexWz4TU)
- **Medium**: [Read Article](https://medium.com/@anilmatcha/clipanything-free-ai-video-editor-in-python-tutorial-526f7a972829)

## Features

- **Natural Language Prompts** - Describe the moment you want in plain English
- **Multimodal Analysis** - Understands visual, audio, and sentiment cues
- **Smart Scene Detection** - Identifies objects, actions, emotions, and text
- **Virality Scoring** - Rates each scene for potential engagement
- **Customizable Clips** - Tailor output to your exact specifications

## How It Works

```
Video Input → AI Analysis → Prompt Matching → Clip Extraction → Output Video
```

### Advanced Video Analysis

The AI evaluates every frame of your video:
- **Visual**: Objects, scenes, actions, faces
- **Audio**: Speech, music, sound effects
- **Sentiment**: Emotions, reactions, tone
- **Text**: On-screen text and captions

### Prompt-Based Clipping

Simply describe what you want:
- "Find all the funny moments"
- "Clip when someone scores a goal"
- "Extract the product reveal"
- "Get the emotional reaction shots"

## Demo

**Input Video**: [YouTube Example](https://www.youtube.com/watch?v=U9mJuUkhUzk)

**Output Video**: [See Result](https://github.com/SamurAIGPT/ClipAnything/blob/main/edited_output.mp4)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/SamurAIGPT/Clip-Anything.git
cd Clip-Anything

# Install dependencies
pip install -r requirements.txt

# Run the clipper
python clip_anything.py --video input.mp4 --prompt "your prompt here"
```

## API Alternative

Want production-ready clipping at scale? Use the [Vadoo AI Clipping API](https://docs.vadoo.tv/docs/guide/create-ai-clips):

```python
import requests

response = requests.post(
    "https://viralapi.vadoo.tv/api/create_clips",
    headers={"X-API-KEY": "your_api_key"},
    json={
        "video_url": "https://example.com/video.mp4",
        "prompt": "highlight moments"
    }
)
```

## Use Cases

| Use Case | Example Prompt |
|----------|----------------|
| Sports Highlights | "Extract all scoring plays" |
| Podcast Clips | "Find the most insightful moments" |
| Travel Vlogs | "Clip the scenic views" |
| Tutorials | "Get the key demonstration steps" |
| Interviews | "Find emotional reactions" |

## Tech Stack

| Component | Purpose |
|-----------|---------|
| GPT-4V | Visual understanding |
| Whisper | Audio transcription |
| FFmpeg | Video processing |
| OpenCV | Frame analysis |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Follow for Updates

- [Anil Chandra Naidu Matcha](https://twitter.com/matchaman11)
- [Ankur Singh](https://twitter.com/ankur_maker)

## Related Projects

- [AI-Youtube-Shorts-Generator](https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator) - Auto-generate YouTube Shorts
- [Text-To-Video-AI](https://github.com/SamurAIGPT/Text-To-Video-AI) - Generate videos from text
- [ai-creator-academy](https://github.com/Anil-matcha/ai-creator-academy) — free curriculum teaching creators how to monetize AI-clipped video content

## License

MIT License - see [LICENSE](LICENSE) for details.
