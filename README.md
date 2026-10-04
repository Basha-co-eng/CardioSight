# 🫀 CardioSight | Uncertainty-Aware Explainable Federated ECG Platform

[![Public Web App](https://img.shields.io/badge/Web_App-Live_Preview-38bdf8?style=for-the-badge&logo=google-chrome)](https://md-sami-7.github.io/CardioSight/)
[![PWA Ready](https://img.shields.io/badge/PWA-iOS_%26_Android_Ready-10b981?style=for-the-badge&logo=pwa)](https://md-sami-7.github.io/CardioSight/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**CardioSight** is a clinical decision-support web and mobile application powered by **Federated Learning**, **Explainable Artificial Intelligence (Grad-CAM & SHAP)**, and **Monte Carlo Dropout Uncertainty Quantification**.

---

## 🌟 Key Features

1. **📱 Cross-Platform (Mobile & Desktop):**
   - Fully responsive for Android, iOS (PWA standalone app installable on home screen), tablets, laptops, and wide desktop monitors.
   - Touch-optimized bottom navigation on mobile devices.
2. **⚡ Real-Time 12-Lead ECG Waveform Canvas:**
   - 25 mm/s medical millimeter paper grid with lead switching (Lead I, II, III, aVR, aVL, aVF, V1-V6, 12-Lead Grid).
   - Real-time physiological metrics: Heart Rate (bpm), PR interval (ms), QRS duration (ms), QT/QTc (ms), Mean RR.
3. **🔥 1D Grad-CAM Saliency Overlay:**
   - Live color-coded heatmaps highlighting high-attention cardiac wave segments (P-wave, delta wave, wide QRS, ST changes, T-wave).
4. **📊 SHAP Feature Attribution:**
   - Interactive bar chart & beeswarm visualization on cardiological fiducial metrics.
5. **🔬 Monte Carlo Dropout Uncertainty Quantification:**
   - 30 stochastic passes evaluating predictive entropy, mutual information, and epistemic variance with clinical safety badges.
6. **🤝 Inter-Hospital Explanation Consistency:**
   - Pairwise cosine similarity matrix across 5 decentralized clinical hospital nodes (*Chapman-Shaoxing, CPSC-2018, Georgia, Ningbo, PTB-XL*).
7. **🩺 Automated Clinical Reasoning & Precautions Generator:**
   - Automatically generates medical reports explaining *why* the condition occurred (pathophysiology/etiology), clinical risk stratification, and patient guidelines.
   - Downloadable formatted `.md` / printable PDF medical report.
8. **🌐 Federated Optimization Comparison:**
   - Evaluates FedAdam, FedProx ($\mu=0.001$), FedAvg, and Centralized baselines with convergence curves and radar charts.

---

## 🚀 Live Demo & Installation

- **Live Web App:** [https://md-sami-7.github.io/CardioSight/](https://md-sami-7.github.io/CardioSight/)
- **Mobile Installation (PWA):** Open in Safari (iOS) or Chrome (Android) and tap **"Add to Home Screen"** to install CardioSight as a native mobile app!

---

## 👥 Authors & Academic Mentorship
- **Syed Mohammad Samiul Ahmed**
- **Shaik Mahammad Galeeb**
- **Shaik Khasim Basha**
- **Shaik Rehaman**

**Project Guide:**
- **Dr. Lalitha Kumari P**  
- School of Computer Science and Engineering (SCOPE), **VIT-AP University**

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
