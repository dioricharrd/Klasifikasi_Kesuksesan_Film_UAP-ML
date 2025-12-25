<p align="center">
  <h1 align="center">🎬 Film Success Prediction</h1>
  <p align="center">
    Machine Learning-powered web application for predicting movie success and generating film synopses
    <br />
    <strong>Built with Flask · Scikit-learn · TF-IDF</strong>
  </p>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#deployment">Deployment</a> •
  <a href="#tech-stack">Tech Stack</a>
</p>

---

## 📖 About

**Film Success Prediction** adalah aplikasi web berbasis Machine Learning yang dapat memprediksi kesuksesan sebuah film berdasarkan berbagai fitur seperti budget, genre, dan deskripsi film. Aplikasi ini menggunakan model Logistic Regression yang telah dilatih pada dataset TMDB 5000 Movies.

### 🎯 Key Features

- **🤖 AI Synopsis Generator**: Generate film synopsis otomatis berdasarkan genre
- **📊 Success Prediction**: Prediksi tingkat kesuksesan film (ROI-based)
- **🎭 Multi-Genre Support**: Mendukung berbagai genre film (Action, Drama, Comedy, dll)
- **💡 Smart Recommendations**: Rekomendasi film sukses berdasarkan genre yang dipilih
- **📈 Probability Score**: Menampilkan confidence score dari prediksi
- **🎨 Modern UI**: Antarmuka yang responsive dan user-friendly

---

## ✨ Features

### 1. Synopsis Generation
Generate deskripsi film secara otomatis berdasarkan genre yang dipilih. AI akan membuat synopsis yang menarik dan sesuai dengan karakteristik genre.

### 2. Success Prediction
Prediksi apakah film akan sukses atau tidak berdasarkan:
- Budget film
- Genre
- Deskripsi/Overview
- Popularity metrics
- Vote statistics

### 3. Film Recommendations
Dapatkan rekomendasi 5 film sukses dari genre yang sama dengan ROI tertinggi.

---

## 🚀 Demo

**Live Demo**: `Coming Soon`

### Screenshots

**Main Interface**
```
┌─────────────────────────────────────────────┐
│  🎬 Film Success Prediction                 │
│                                              │
│  Generate Synopsis                           │
│  ┌──────────────────────────────────┐       │
│  │ Select Genre: [Action ▼]         │       │
│  │ [Generate Synopsis]               │       │
│  └──────────────────────────────────┘       │
│                                              │
│  Predict Success                             │
│  ┌──────────────────────────────────┐       │
│  │ Budget: $100,000,000             │       │
│  │ Genres: Action, Sci-Fi            │       │
│  │ Overview: [...]                   │       │
│  │ [Predict]                         │       │
│  └──────────────────────────────────┘       │
└─────────────────────────────────────────────┘
```

---

## 📋 Requirements

- Python 3.10+
- pip (Python package manager)
- 512 MB RAM minimum (2 GB recommended)
- 15 MB free disk space

### Python Dependencies

```
Flask==3.1.2
scikit-learn==1.8.0
pandas==2.3.3
numpy==2.4.0
joblib==1.5.3
gunicorn==23.0.0
```

---

## 🛠️ Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/USERNAME/film-prediction.git
   cd film-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app_multi_model.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

---

## 📱 Usage

### Generate Synopsis

1. Pilih genre film dari dropdown
2. Klik tombol **Generate Synopsis**
3. Synopsis akan muncul di text area

### Predict Film Success

1. Masukkan **Budget** film (dalam USD)
2. Pilih **Genre** (bisa lebih dari satu, pisahkan dengan koma)
3. Masukkan **Overview/Synopsis** film
4. Isi fitur tambahan:
   - Popularity
   - Vote Average
   - Vote Count
   - Runtime (menit)
5. Klik **Predict**

### Hasil Prediksi

Aplikasi akan menampilkan:
- ✅ **Success** atau ❌ **Not Success**
- 📊 Probability score (confidence level)
- 🎬 5 Rekomendasi film sukses dari genre yang sama

---

## 🌐 Deployment

### Deploy to PythonAnywhere

1. **Upload files to PythonAnywhere**
   ```bash
   # Via Git
   git clone https://github.com/USERNAME/film-prediction.git
   cd film-prediction
   ```

2. **Setup virtual environment**
   ```bash
   mkvirtualenv --python=/usr/bin/python3.10 filmenv
   pip install -r requirements.txt
   ```

3. **Configure WSGI**
   
   Edit `/var/www/username_pythonanywhere_com_wsgi.py`:
   ```python
   import sys
   path = '/home/username/film-prediction'
   if path not in sys.path:
       sys.path.insert(0, path)
   
   activate_this = '/home/username/.virtualenvs/filmenv/bin/activate_this.py'
   with open(activate_this) as f:
       exec(f.read(), {'__file__': activate_this})
   
   from app_multi_model import app as application
   ```

4. **Set virtualenv path**
   ```
   /home/username/.virtualenvs/filmenv
   ```

5. **Reload and visit**
   ```
   https://username.pythonanywhere.com
   ```

### Deploy to Heroku

```bash
# Login to Heroku
heroku login

# Create new app
heroku create film-prediction-app

# Deploy
git push heroku main

# Open app
heroku open
```

---

## 🧪 Model Information

### Dataset
- **Source**: TMDB 5000 Movies Dataset
- **Size**: 4,803 films
- **Features**: 20+ attributes (budget, revenue, genres, overview, etc.)

### Model Performance
- **Algorithm**: Logistic Regression
- **Features**: TF-IDF (synopsis) + Numerical (budget, popularity, etc.)
- **Accuracy**: ~85% on test set
- **ROI Threshold**: 1.0 (film considered successful if ROI > 1.0)

### Feature Engineering
- TF-IDF vectorization untuk text (overview)
- Standard scaling untuk numerical features
- Genre encoding
- ROI calculation: `(revenue - budget) / budget`

---

## 🛠️ Tech Stack

### Backend
- **Flask** - Python web framework
- **Scikit-learn** - Machine Learning library
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Joblib** - Model serialization

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling (Bootstrap)
- **JavaScript** - Interactivity
- **AJAX** - Asynchronous requests

### Deployment
- **Gunicorn** - WSGI HTTP Server
- **PythonAnywhere** - Cloud hosting
- **Git** - Version control

---

## 📁 Project Structure

```
film-prediction/
├── app_multi_model.py          # Main Flask application
├── requirements.txt             # Python dependencies
├── Procfile                     # Heroku deployment config
├── runtime.txt                  # Python version
├── success_model.pkl            # Trained ML model
├── success_tfidf.pkl            # TF-IDF vectorizer
├── success_scaler.pkl           # Feature scaler
├── success_feature_info.json   # Feature metadata
├── tmdb_5000_movies.csv        # Dataset
└── templates/
    └── multi_model.html        # Main UI template
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add comments for complex logic
- Update documentation for new features
- Test thoroughly before submitting PR

---

## 🐛 Known Issues

- Large model files (~8 MB) might cause slow initial loading
- Limited to TMDB dataset genres
- Requires sufficient disk space for model files

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- TMDB for providing the movie dataset
- Scikit-learn community for excellent ML tools
- Flask team for the wonderful web framework
- All contributors and supporters

---

## 📧 Contact

- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your Name](https://linkedin.com/in/yourprofile)

---

<p align="center">Made with ❤️ and ☕</p>
<p align="center">⭐ Star this repo if you find it helpful!</p>
