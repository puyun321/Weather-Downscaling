# Welcome to the Jhuoshuei Watershed Dataset Repository! 🌊🌿

Dive into the dynamic world of data science and environmental research with this repository, which houses datasets and models for analyzing key factors in the Jhuoshuei River watershed. Here's what you'll find inside:

## 📂 Folder Structure & Data Highlights

- **Image datasets:**  
  Explore visual representations of essential factors from towns within the Jhuoshuei River watershed. These include:  
  - **Temperature (T)** 🌡️  
  - **Relative Humidity (RH)** 💧  
  These images are generated using:  
  - `T/preprocessing/generate_image_data.py` for temperature  
  - `RH/preprocessing/generate_image_data.py` for relative humidity

- **Code Datasets:**  
  Processed code datasets are extracted by:
  - `T/preprocessing/autoencoder.py` for temperature  
  - `RH/preprocessing/autoencoder.py` for relative humidity  

## 🧠 Models and Benchmarks

Unleash the power of deep learning with our specialized models tailored to temperature and humidity analysis:

- **For Temperature (T):**  
  - **CAE-LSTMT Model**: `T/CAE-LSTMT.py`  
  - **Benchmark Model**: `T/benchmark.py`

- **For Relative Humidity (RH):**  
  - **CAE-LSTMT Model**: `RH/CAE-LSTMT.py`  
  - **Benchmark Model**: `RH/benchmark.py`

---

## Why It Matters 🚀

This repository isn’t just about code and data—it’s about understanding and predicting the intricate hydrological and atmospheric phenomena of the Jhuoshuei River watershed. Whether you're an environmental researcher, data scientist, or enthusiast, there's something here to explore, analyze, and innovate.

Happy exploring! 🌟
