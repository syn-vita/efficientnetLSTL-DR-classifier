// Simple ONNX runtime web-based inference helper (CPU)
// Note: You need to add onnxruntime-web to dependencies and place .onnx files in public/models

export async function runOnnx(modelUrl, imageBitmap) {
  // Use global ort loaded via script tag from /ort
  const ort = window.ort || globalThis.ort;
  if (!ort) throw new Error('ONNX Runtime not loaded.');
  // Ensure ORT can find WASM assets when hosted under /ort
  if (ort.env && ort.env.wasm) {
    ort.env.wasm.wasmPaths = '/ort/';
  }

  // Preprocess: resize to 224x224, normalize to ImageNet
  const size = 224;
  let canvas;
  if (typeof OffscreenCanvas !== 'undefined') {
    canvas = new OffscreenCanvas(size, size);
  } else {
    canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
  }
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imageBitmap, 0, 0, size, size);
  const { data } = ctx.getImageData(0, 0, size, size);

  // Convert RGBA to NCHW float32 normalized
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const chw = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    chw[i] = (r - mean[0]) / std[0]; // R
    chw[size * size + i] = (g - mean[1]) / std[1]; // G
    chw[2 * size * size + i] = (b - mean[2]) / std[2]; // B
  }

  const tensor = new ort.Tensor('float32', chw, [1, 3, size, size]);
  // Prefer wasm backend; disable simd/threaded auto if environment fails to load them
  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
  });

  // Try to guess the input name (common: 'input', 'images', 'input_0')
  const candidateInputs = [
    ...(session.inputNames || []),
    'input', 'images', 'input_0', 'x'
  ];
  const inputName = candidateInputs.find(Boolean);
  if (!inputName) {
    throw new Error('Model input name could not be determined.');
  }
  const inputs = { [inputName]: tensor };
  const outputMap = await session.run(inputs);
  const outputName = Object.keys(outputMap)[0];
  const logits = outputMap[outputName].data;

  // Softmax
  const maxLogit = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(v => v / sumExps);
  const topIdx = probs.indexOf(Math.max(...probs));

  return { classIndex: topIdx, probabilities: probs };
}
