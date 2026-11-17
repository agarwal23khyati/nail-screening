# 💅 AI-Based Nail Screening System

An AI-powered nail image analysis and preliminary screening system that uses computer vision and deep learning to identify nail regions, analyse visual characteristics, and flag patterns that may require further medical attention.

## 🌟 Project Overview

The AI-Based Nail Screening System is designed to analyse nail images using a U-Net-based image segmentation model and image-processing techniques.

The system first verifies image quality and identifies the nail region before analysing its visual characteristics. It can flag certain colour patterns associated with possible health-related conditions.

The project aims to make preliminary visual screening more accessible through an easy-to-use web interface.

## ✨ Key Features

- **AI-Based Nail Segmentation:** Uses a trained U-Net model to identify and isolate nail regions.
- **Image Quality Assessment:** Checks blur, glare, and brightness before processing.
- **Nail Verification:** Uses geometric and appearance-based checks to verify the detected region.
- **Colour Analysis:** Extracts RGB, HSV, and CIELAB colour features.
- **Preliminary Screening:** Identifies visual patterns such as pale, yellowish, and bluish nail appearances.
- **Retake Detection:** Rejects unsuitable images instead of generating unreliable screening results.
- **Web Interface:** Provides an interactive interface for uploading and analysing nail images.

## 🛠️ Technology Stack

| Component            | Technology            |
| -------------------- | --------------------- |
| Programming Language | Python                |
| Backend              | Flask                 |
| Deep Learning        | PyTorch               |
| Image Segmentation   | U-Net                 |
| Image Processing     | OpenCV                |
| Numerical Processing | NumPy, SciPy          |
| Image Analysis       | Scikit-image          |
| Machine Learning     | Scikit-learn          |
| Frontend             | HTML, CSS, JavaScript |

## ⚙️ Project Architecture

The system follows a sequential processing pipeline:

1. **Image Upload:** User uploads a nail image through the web interface.
2. **Quality Assessment:** Image is checked for blur, excessive glare, and insufficient brightness.
3. **Nail Segmentation:** The trained U-Net model identifies the nail region.
4. **Nail Verification:** Geometry, confidence, stability, contrast, and texture checks validate the detected region.
5. **Feature Extraction:** Colour characteristics are extracted from the verified nail region.
6. **Screening:** The extracted features are evaluated to identify visual patterns.
7. **Result Generation:** The system displays the screening output and relevant visual information.

## 📁 Project Structure

```text
nail_backend/
│
├── app.py
├── screening.py
├── nail_gate.py
├── train_gate.py
├── requirements.txt
├── RUN.md
│
├── templates/
│   └── index.html
│
└── nailsegmenter_project/
    ├── best_unet.pth
    ├── nail_segmenter.py
    ├── run.ipynb
    ├── photos/
    └── output/
```

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/agarwal23khyati/nail-screening.git
```

### 2. Navigate to the Project Folder

```bash
cd nail-screening
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python app.py
```

### 7. Open the Application

Visit:

```text
http://localhost:5000
```

## 🧠 AI and Screening Methodology

The project combines deep learning and traditional image processing.

- U-Net-based segmentation isolates the nail region.
- Multi-scale and flip test-time augmentation help improve segmentation stability.
- Nail verification prevents incorrect regions from being analysed.
- Colour features are extracted using RGB, HSV, and CIELAB colour spaces.
- Rule-based screening identifies selected visual patterns.

An optional logistic-regression verification model can also be trained using labelled good and bad nail-region examples.

## 🔮 Future Scope

- Expansion of the nail image dataset.
- Improved model accuracy across different lighting conditions and skin backgrounds.
- Integration of additional nail-related visual indicators.
- Mobile application development.
- Improved screening through larger and more diverse datasets.
- Clinical validation and evaluation with qualified healthcare professionals.

## ⚠️ Disclaimer

This project is intended for educational and research purposes. It is a preliminary visual screening tool and is not a substitute for professional medical diagnosis, laboratory testing, or consultation with a qualified healthcare professional.

Visual nail characteristics alone cannot confirm any medical condition.

##
