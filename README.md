# 🎥 Hand Gesture Zoom Application

An interactive computer vision application that allows you to zoom into images using hand gestures detected through your camera. Control zoom levels with simple pinch gestures using your thumb and index finger!

## ✨ Features

- **Hand Gesture Control**: Zoom in/out using pinch gestures (thumb + index finger)
- **Multi-Camera Support**: Automatically detects and allows selection between multiple cameras
- **Image Navigation**: Browse through multiple images in your directory
- **Real-time Processing**: Smooth zoom effects with FPS monitoring
- **Interactive UI**: On-screen controls and visual feedback
- **Zoom Smoothing**: Fluid zoom transitions for better user experience

## 🛠️ Requirements

### Hardware
- Camera (built-in webcam or USB camera)
- Computer with Python support

### Software Dependencies
```bash
pip install opencv-python
pip install numpy
pip install cvzone
```

## 📁 Project Structure

```
hand-gesture-zoom/
├── main.py           # Main application file
├── README.md         # This file
├── LICENSE           # MIT License
├── .gitignore        # Git ignore file
└── [image files]     # Your images (.jpg, .jpeg, .png, .bmp, .tiff)
```

## 🚀 Installation & Setup

1. **Clone or download the repository**
2. **Install dependencies**:
   ```bash
   pip install opencv-python numpy cvzone
   ```
3. **Add images to the project directory**
   - Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
   - Place image files in the same directory as `main.py`

4. **Run the application**:
   ```bash
   python main.py
   ```

## 🎮 How to Use

### Camera Selection
- On first run, the application will detect all available cameras
- Choose your preferred camera from the list
- Use built-in camera (index 0) or external USB cameras

### Hand Gestures
- **Zoom Control**: Make a pinch gesture with thumb and index finger
  - Move fingers apart = Zoom In
  - Move fingers together = Zoom Out
- **Single Hand Only**: Use one hand for gesture recognition

### Keyboard Controls
| Key | Action |
|-----|--------|
| `q` | Quit application |
| `n` | Next image |
| `p` | Previous image |
| `r` | Reset zoom level |
| `c` | Change camera |
| `i` | Toggle instructions display |
| `f` | Toggle FPS counter |

### Visual Indicators
- **Green line**: Shows distance between thumb and index finger
- **Blue circles**: Highlight detected fingertips
- **Green circle**: Shows zoom center point
- **On-screen info**: Camera info, image info, zoom level, FPS

## 🎯 Gesture Recognition Details

The application uses CVZone's HandTracking module to:
- Detect hand landmarks in real-time
- Recognize specific finger configurations
- Calculate distance between thumb and index finger
- Apply smooth zoom transformations

**Optimal Usage**:
- Ensure good lighting conditions
- Keep hand visible in camera frame
- Use clear pinch gestures for best recognition
- Maintain moderate distance from camera

## ⚙️ Configuration Options

### Camera Settings
- Resolution: 1280x720 (adjustable in code)
- FPS: 30 (adjustable in code)
- Auto-detection of available cameras

### Zoom Parameters
- Zoom range: -200 to +300
- Smoothing factor: 0.7 (adjustable)
- Sensitivity: 1.5 (adjustable)

### Customization
You can modify these parameters in the `__init__` method of the `HandGestureZoom` class:
```python
self.min_scale = -200        # Minimum zoom level
self.max_scale = 300         # Maximum zoom level
self.zoom_smoothing = 0.7    # Smoothing factor (0-1)
```

## 🔧 Troubleshooting

### Common Issues

**Camera Not Detected**
- Ensure camera is not being used by another application
- Check camera permissions in system settings
- Try disconnecting and reconnecting USB cameras
- Run application as administrator if needed

**Poor Hand Detection**
- Ensure adequate lighting
- Keep hand clearly visible in frame
- Clean camera lens
- Adjust `detectionCon` parameter (default: 0.7)

**No Images Found**
- Place supported image files in the same directory as `main.py`
- Check file extensions are supported
- Ensure files are not corrupted

**Performance Issues**
- Close other camera applications
- Reduce camera resolution in code
- Lower FPS settings
- Use a more powerful computer

## 🏗️ Advanced Usage

### Command Line Arguments
```bash
python main.py [camera_index]
```
Example: `python main.py 1` to use camera index 1

### Adding New Features
The code is structured for easy extension:
- `detect_zoom_gesture()`: Modify gesture recognition
- `apply_zoom_effect()`: Change zoom behavior
- `draw_ui()`: Customize user interface
- `load_images()`: Add support for more file formats

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CVZone**: For excellent hand tracking capabilities
- **OpenCV**: For computer vision functionality
- **NumPy**: For efficient array operations

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Ensure all dependencies are installed correctly
3. Verify camera functionality with other applications
4. Check Python and library versions

## 🔄 Version History

- **v1.0**: Initial release with basic zoom functionality
- **v1.1**: Added multi-camera support and enhanced UI
- **v1.2**: Improved gesture recognition and added keyboard controls

---

**Enjoy zooming with your hands! 👋📸**
