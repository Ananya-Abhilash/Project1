# 🎵 Music Genre Classifier using CNN and MFCC

This project is a Convolutional Neural Network (CNN) based music genre classifier that takes `.wav` audio files as input and predicts one of 10 possible genres. It uses Mel Frequency Cepstral Coefficients (MFCC) as the primary audio feature representation.

---

## 📚 Dataset

We use the **GTZAN Genre Collection** dataset which contains:

- 1000 audio tracks of 30 seconds each
- 10 genres: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`
- Each genre has 100 audio files

🔗 **Download dataset from**:  
[Kaggle: GTZAN Genre Collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download)

---

## 🏗 Model Architecture

The model is a Sequential CNN built with the following layers:

- `Conv2D` → `BatchNormalization` → `MaxPooling2D` → `Dropout`
- Repeated three times with increasing filters (32, 64, 128)
- `Flatten` → `Dense(128)` → `Dropout` → `Dense(num_classes)` with softmax

It expects input of shape `(65, 40, 1)`, which corresponds to padded/truncated MFCCs.

---

## 🧪 Feature Extraction

Audio preprocessing is done using `librosa`:

- Extract 40 MFCCs per frame
- Use a frame length (`n_fft`) of 2048 and hop length of 512
- Pad/truncate sequences to make uniform shape: `(65, 40)`
- Reshape for CNN input: `(65, 40, 1)`

---

## 🧠 Training

* Loss function: `categorical_crossentropy`
* Optimizer: `Adam`
* Metrics: `accuracy`
* Training set: 80% of GTZAN
* Test accuracy: \~85–88% depending on tuning and preprocessing

---

## 📁 Project Contents

* `genre_classifier.ipynb`: Full code (preprocessing, training, prediction)
* `157747__flick3r__fairytale.wav`: Sample audio clip for testing
* `README.md`: This file

---

## 💾 Installation

Install dependencies using pip:

```bash
pip install tensorflow librosa numpy matplotlib scikit-learn
```

---

## 🚀 How to Run

1. Download and extract the GTZAN dataset
2. Run the notebook `genre_classifier.ipynb` to:

   * Extract features
   * Train the CNN
   * Evaluate performance
   * Predict genre for a new clip
3. Optionally, test with your own `.wav` files

---

## 🎯 Future Improvements

* Add support for live microphone input
* Use mel-spectrograms or chroma features
* Deploy as a web app using Streamlit or Flask
* Try advanced models like CRNN or Transformers

---

## 📎 Example Audio Clip

For quick testing, you can download a sample `.wav` file. Sample `.wav` file given.

---

## 📜 License

This project is for educational purposes.
The GTZAN dataset is used under fair use for academic research.

---

## 🤖 Credits

Built using:

* [TensorFlow/Keras](https://www.tensorflow.org/)
* [Librosa](https://librosa.org/)
* [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download)
