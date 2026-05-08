// Model configurations
export const modelConfig = {
  lstl: {
    name: 'EfficientNet-B0 + LSTL',
    shortName: 'LSTL',
    badge: 'Recommended',
    description:
      'The thesis-highlighted EfficientNet-B0 variant integrates a Local-Global Spatially-Aware Transformer Layer for the primary research-facing experience.',
    architecture: 'EfficientNet-B0 + LSTL',
    parameters: '4.05M',
    accuracy: 'Balanced accuracy (val, best epoch, fold 4): 67.19%',
    gflops: 0.420,
    modelPath: '/models/efficientnet_b0_lstl_clean_fold_4.onnx',
    bestFold: 4
  },
  baseline: {
    name: 'EfficientNet-B0 Baseline',
    shortName: 'Baseline',
    badge: 'Reference',
    description:
      'The standard EfficientNet-B0 implementation serves as the architectural baseline for thesis comparison.',
    architecture: 'EfficientNet-B0',
    parameters: '4.01M',
    accuracy: 'Balanced accuracy (val, best epoch, fold 3): 70.17%',
    gflops: 0.414,
    modelPath: '/models/efficientnet_b0_clean_fold_3.onnx',
    bestFold: 3
  },
  cbam: {
    name: 'EfficientNet-B0 + CBAM',
    shortName: 'CBAM',
    badge: 'Alternative',
    description:
      'The CBAM-enhanced EfficientNet-B0 variant provides an attention-based comparison point within the thesis workflow.',
    architecture: 'EfficientNet-B0 + CBAM',
    parameters: '4.22M',
    accuracy: 'Balanced accuracy (val, best epoch, fold 4): 68.77%',
    gflops: 0.414,
    modelPath: '/models/efficientnet_b0_cbam_clean_fold_4.onnx',
    bestFold: 4
  }
}

// DR classification classes
export const drClasses = [
  {
    id: 0,
    name: 'No DR',
    description: 'No signs of diabetic retinopathy detected.',
    recommendation: 'Continue regular eye examinations. Maintain good blood sugar control.',
    severity: 'None'
  },
  {
    id: 1,
    name: 'Mild NPDR',
    description: 'Mild non-proliferative diabetic retinopathy with microaneurysms.',
    recommendation: 'Annual comprehensive eye examination recommended. Monitor blood glucose levels.',
    severity: 'Mild'
  },
  {
    id: 2,
    name: 'Moderate NPDR',
    description: 'Moderate non-proliferative diabetic retinopathy with retinal hemorrhages.',
    recommendation: 'Follow-up in 6-12 months. Consider more frequent monitoring.',
    severity: 'Moderate'
  },
  {
    id: 3,
    name: 'Severe NPDR',
    description: 'Severe non-proliferative diabetic retinopathy with extensive retinal changes.',
    recommendation: 'Urgent ophthalmologic referral within 2-4 weeks. Consider treatment options.',
    severity: 'Severe'
  },
  {
    id: 4,
    name: 'Proliferative DR',
    description: 'Proliferative diabetic retinopathy with neovascularization.',
    recommendation: 'Immediate ophthalmologic referral. Requires prompt treatment to prevent vision loss.',
    severity: 'Very Severe'
  }
]

// Customization settings
export const appConfig = {
  title: 'Thesis: Integrating LSTL with EfficientNetB0 for Diabetic Retinopathy Classification',
  compactTitle: 'Thesis: LSTL + EfficientNetB0 for DR Classification',
  subtitle: 'Research-use screening interface',
  maxFileSize: 10 * 1024 * 1024, // 10MB
  acceptedFormats: ['image/jpeg', 'image/jpg', 'image/png'],
  theme: {
    primary: '#1098ad',
    success: '#22c55e',
    warning: '#f59e0b',
    danger: '#ef4444'
  }
}
