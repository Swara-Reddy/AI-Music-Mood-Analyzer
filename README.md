# 🎵 MoodWave - AI Music Mood Analyzer

An intelligent music analysis platform that predicts the emotional mood of songs using Machine Learning and audio signal processing.

MoodWave analyzes uploaded audio files, extracts advanced audio features, predicts the song's mood, visualizes audio patterns, and recommends similar tracks based on musical characteristics.

---

## 🚀 Live Demo

https://ai-music-mood-analyzer-b5m63qpvy8gvthm8m3vst8.streamlit.app

## 🚀 Features

### 🎯 Mood Prediction
Predicts music into one of the following categories:

- 😊 Happy
- 😢 Sad
- ❤️ Romantic
- 🎭 Dramatic
- 🔥 Aggressive

---

### 📊 Confidence Analysis

Displays prediction confidence for all moods using interactive progress bars.

Example:

Happy: 82%

Sad: 10%

Romantic: 5%

Dramatic: 2%

Aggressive: 1%

---

### 🎵 Audio Feature Extraction

Extracts advanced music features using Librosa:

- MFCC (Mel-Frequency Cepstral Coefficients)
- Chroma Features
- Tempo (BPM)
- RMS Energy
- Spectral Centroid
- Spectral Bandwidth
- Zero Crossing Rate

---

### 📈 Interactive Visualizations

#### 🌊 Waveform Analysis

Visual representation of audio amplitude over time.

#### 🎼 Spectrogram Analysis

Frequency distribution visualization of the uploaded audio.

---

### 🎧 Song Recommendation System

Recommends similar songs using:

- Feature-based similarity
- Cosine Similarity Matching
- Content-Based Filtering

---

### 🧠 AI Interpretation

Generates human-readable insights such as:

- Relaxed Tempo
- High Energy Track
- Bright Sound Signature
- Emotional Characteristics

---

### 🎨 Dynamic Mood Themes

The application's visual theme automatically changes according to the predicted mood:

| Mood | Theme |
|--------|--------|
| Happy | 🟡 Yellow |
| Sad | 🔵 Blue |
| Romantic | 🌸 Pink |
| Aggressive | 🔴 Red |
| Dramatic | 🟣 Purple |

---

## 🛠️ Tech Stack

### Machine Learning

- Scikit-Learn
- Random Forest Classifier
- Feature Scaling
- Train-Test Split
- PCA

### Audio Processing

- Librosa
- NumPy

### Data Analysis

- Pandas

### Visualization

- Matplotlib

### Web Application

- Streamlit

---

## 📂 Dataset

Custom dataset containing:

- 500 Audio Samples
- 5 Mood Categories
- WAV Audio Format

Dataset Distribution:

| Mood | Samples |
|--------|----------|
| Happy | 100 |
| Sad | 100 |
| Romantic | 100 |
| Dramatic | 100 |
| Aggressive | 100 |

---

## 🏆 Model Performance

### Random Forest Classifier

Accuracy Achieved:

```text
90%
```

Evaluation Metrics:

- Accuracy Score
- Classification Report
- Confusion Matrix

---

## 📸 Application Preview

### Mood Prediction Dashboard

- Upload Song
- Predict Mood
- View Confidence Scores
- Analyze Audio Features

### Visual Analytics

- Waveform Visualization
- Spectrogram Visualization
- Music Statistics

### Recommendations

- Similar Songs
- Similarity Scores
- Mood-Based Suggestions

---

## 📂 Project Structure

```text
music-ai-project/
│
├── app/
│   └── app.py
│
├── data/
│   └── music_features_dataset.csv
│
├── models/
│   ├── music_mood_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   └── day1_audio_analysis.ipynb
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/MoodWave.git
```

Move into the project folder:

```bash
cd MoodWave
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/app.py
```

---

## 🎓 Key Concepts Implemented

- Machine Learning Classification
- Audio Signal Processing
- Feature Engineering
- Recommendation Systems
- Data Visualization
- Content-Based Filtering
- Model Deployment
- Interactive Web Applications

---

## 📌 Future Improvements

- Spotify API Integration
- Real-Time Mood Detection
- Playlist Generation
- Deep Learning Models
- Genre Classification
- Multi-Mood Prediction
- Music Embeddings
- User Preference Learning

---

## 👨‍💻 Author

Swara Reddy

B.Tech CSE (AI & ML)

Machine Learning • Data Science • Audio Intelligence

---

⭐ If you found this project interesting, consider giving it a star!
