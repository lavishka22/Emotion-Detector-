# Emotion Detector 😊

An AI-powered real-time emotion detection system that analyzes facial expressions using computer vision and deep learning to classify emotions accurately.

## 🎯 Features

- ✨ Real-time facial emotion classification
- 😄 Multi-emotion support (Happy, Sad, Angry, Surprised, Neutral, Fear, Disgust)
- 🚀 High-performance neural network architecture
- 📷 Webcam and image file support
- 🎬 Video processing capabilities
- 📊 Confidence scores for predictions
- ⚡ Optimized for CPU and GPU

## 📋 Requirements

- Python 3.8 or higher
- OpenCV
- TensorFlow / Keras
- NumPy
- Pandas

## 🔧 Installation

Clone the repository:
```bash
git clone https://github.com/lavishka22/Emotion-Detector-.git
cd Emotion-Detector-
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Real-time Webcam Detection
```python
from emotion_detector import EmotionDetector

detector = EmotionDetector()
detector.detect_from_webcam()
```

### Image File Detection
```python
from emotion_detector import EmotionDetector

detector = EmotionDetector()
emotions = detector.detect_from_image('path/to/image.jpg')
print(emotions)
```

### Video File Detection
```python
from emotion_detector import EmotionDetector

detector = EmotionDetector()
detector.detect_from_video('path/to/video.mp4')
```

## 📚 Model Architecture

- **Base Model**: Convolutional Neural Network (CNN)
- **Input Size**: 48x48 grayscale images
- **Output**: 7 emotion classes
- **Accuracy**: [Add your model accuracy]
- **Training Dataset**: [Add dataset source]

## 📊 Emotion Classes

| Emotion | Label |
|---------|-------|
| Angry | 0 |
| Disgust | 1 |
| Fear | 2 |
| Happy | 3 |
| Neutral | 4 |
| Sad | 5 |
| Surprised | 6 |

## 📁 Project Structure

```
Emotion-Detector-/
├── README.md
├── requirements.txt
├── emotion_detector.py
├── model/
│   └── emotion_model.h5
├── data/
│   ├── train/
│   └── test/
└── demo.py
```

## 🎬 Demo

Run the interactive demo:
```bash
python demo.py
```

## 🔬 Performance Metrics

- Training Accuracy: [Add your metrics]
- Validation Accuracy: [Add your metrics]
- Test Accuracy: [Add your metrics]

## 💡 How It Works

1. **Face Detection**: Uses cascade classifiers to detect faces in images
2. **Preprocessing**: Converts images to grayscale and resizes to 48x48
3. **Emotion Classification**: Passes preprocessed images through CNN
4. **Prediction**: Returns emotion label with confidence score

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs via Issues
- Submit Pull Requests with improvements
- Suggest new features

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## ✉️ Contact

For questions or suggestions, feel free to reach out or open an issue.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- TensorFlow/Keras team for deep learning framework
- [Add any dataset citations]

---

**Happy Detecting! 🎉**
