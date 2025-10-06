# DR Classification Webapp

An advanced, professional web application for diabetic retinopathy classification using state-of-the-art AI models. Built with Vite, React, and Tailwind CSS, featuring a comprehensive landing page and enhanced user experience designed for academic presentations.

## ✨ Key Features

### 🎯 **Professional Landing Page**
- Engaging hero section with gradient backgrounds and animations
- Feature showcase highlighting AI technology
- Model comparison section with detailed specifications
- Research impact presentation
- Smooth scrolling navigation

### 🧠 **Three Advanced AI Models**
- **Baseline EfficientNet-B0**: Standard architecture for comparison
- **CBAM-Enhanced**: Convolutional Block Attention Module integration
- **LSTL-Optimized**: Lightweight Spatial Temporal Learning implementation

### 🎨 **Enhanced User Interface**
- Modern gradient designs and micro-animations
- Improved visual hierarchy and typography
- Interactive hover effects and transitions
- Responsive design for all screen sizes
- Professional color scheme with blue/indigo gradients

### 📊 **Advanced Analytics Display**
- Real-time classification with confidence visualization
- Animated probability distributions
- Severity level indicators with progress bars
- Clinical recommendations and disclaimers
- Enhanced loading states with animated indicators

### 🔧 **Technical Excellence**
- Fast model inference using ONNX.js
- Drag & drop image upload with preview
- Client-side processing for privacy
- Optimized for thesis presentations

## 🚀 Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Development Server
```bash
npm run dev
```
Visit `http://localhost:3000`

### 3. Build for Production
```bash
npm run build
```

### 4. Deploy to Vercel
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

## Model Integration

### Adding Your Trained Models

1. **Export Best Models**: From your training, identify the best performing fold for each model:
   ```python
   # Find best fold based on validation accuracy
   best_baseline_fold = "efficientnet_b0_baseline_clean_fold_X.pth"
   best_cbam_fold = "efficientnet_b0_cbam_clean_fold_Y.pth" 
   best_lstl_fold = "efficientnet_b0_lstl_clean_fold_Z.pth"
   ```

2. **Convert to ONNX** (recommended for web deployment):
   ```python
   import torch
   import torch.onnx
   
   # Load your best model
   model = EfficientNetB0WithLSTL(num_classes=5)
   model.load_state_dict(torch.load('best_fold.pth'))
   model.eval()
   
   # Export to ONNX
   dummy_input = torch.randn(1, 3, 224, 224)
   torch.onnx.export(model, dummy_input, "model.onnx")
   ```

3. **Place Models**: Put converted models in `public/models/`

### Backend API (Optional)

For server-side inference, create `api/predict.js`:

```javascript
// Example Vercel serverless function
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Handle image upload and model inference
  // Return classification results
}
```

## Customization

### Colors and Styling

Edit `tailwind.config.js` to change the color scheme:

```javascript
theme: {
  extend: {
    colors: {
      primary: {
        500: '#your-color',
        600: '#your-darker-color',
        // ...
      }
    }
  }
}
```

### Model Information

Update `src/config/models.js`:

```javascript
export const modelConfig = {
  baseline: {
    name: 'Your Model Name',
    accuracy: 'XX.X%',
    // ...
  }
}
```

### App Settings

Modify `src/config/models.js`:

```javascript
export const appConfig = {
  title: 'Your App Title',
  maxFileSize: 10 * 1024 * 1024, // 10MB
  // ...
}
```

## File Structure

```
dr-classification-webapp/
├── src/
│   ├── components/
│   │   ├── Header.jsx          # Navigation with model tabs
│   │   ├── ImageUpload.jsx     # Drag & drop upload
│   │   └── ResultDisplay.jsx   # Classification results
│   ├── config/
│   │   └── models.js           # Model and app configuration
│   ├── App.jsx                 # Main application
│   ├── main.jsx               # React entry point
│   └── index.css              # Tailwind styles
├── public/
│   ├── models/                 # Model files (.pth, .onnx)
│   └── medical-icon.svg        # App icon
├── vercel.json                 # Vercel deployment config
└── package.json
```

## Deployment Options

### Vercel (Recommended)
```bash
vercel --prod
```

### Netlify
```bash
npm run build
# Upload dist/ folder to Netlify
```

### Custom Server
```bash
npm run build
# Serve dist/ folder with any static server
```

## Performance Tips

1. **Image Optimization**: Resize images to 224x224 before upload
2. **Model Optimization**: Use ONNX.js or TensorFlow.js for client-side inference
3. **Caching**: Enable CDN caching for model files
4. **Compression**: Use model quantization to reduce file sizes

## Browser Support

- Chrome 88+
- Firefox 78+
- Safari 14+
- Edge 88+

## License

MIT License - feel free to modify and distribute.

## Using best-fold models and real inference

To run real inference in the browser with the best fold per model:

1. Install ONNX Runtime Web
  ```bash
  npm i onnxruntime-web
  ```
2. Export your best folds to ONNX with input [1,3,224,224].
3. Place the .onnx files into `public/models` and ensure filenames match `src/config/models.js` (e.g., `efficientnet_b0_lstl_best_fold.onnx`).
4. Optionally update the `bestFold` field in each model config; the UI displays the selected fold.
5. Start the dev server and upload an image; the app will use ONNX if available, or fall back to a mock prediction otherwise.

Note: If your ONNX model uses different input/output tensor names, update `src/utils/inference.js` accordingly.
