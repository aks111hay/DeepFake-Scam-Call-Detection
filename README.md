# Deepfake Scam Call Detection

##  Overview
This project is designed to detect deepfake scam calls using machine learning models based on Mel-frequency cepstral coefficients (mFCC). Two models were developed and compared:

- **Support Vector Machine (SVM)**: Achieved **95% accuracy**
- **Logistic Regression**: Achieved **80% accuracy**

##  Files Included
- `app.py`: Flask web application to simulate and run the model.
- `fake.wav`: Example of a deepfake scam audio sample.
- `real.wav`: Example of a real (genuine) audio sample.
- `models/`: Folder containing trained models (if applicable).
- `README.md`: Project overview and instructions.
- `deepfake audio detection.ipynb`: Jupyter notebook (Audio analysis and model devlopement).

##  Requirements
- Python 3.x
- Required libraries:
  - `scikit-learn`
  - `Flask`
  - `librosa`
  - `numpy`
  - `soundfile`

##  Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aks111hay/DeepFake-Scam-Call-Detection.git

## install The dependencies
```
pip install -r requirements.txt
```
## run the web application
```
python app.py
```
## Model Comparison

| Model               | Accuracy |
|---------------------|----------|
| SVM (with mFCC)     | 95%      |
| Logistic Regression | 80%      |

## Future Work

- Add support for real-time call classification
- Expand the dataset for improved generalization
- Deploy on a cloud-based platform for broader access

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss the proposed updates.



