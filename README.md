# 📊 Alphabet Soup Charity: Deep Learning Challenge

A binary classification project using TensorFlow and Keras to help **Alphabet Soup**, a nonprofit foundation, better identify which applicants are most likely to use funding effectively. This challenge leverages deep learning to build, evaluate, and optimize a neural network model trained on historical funding data from over 34,000 organizations.

---

## 🔍 Project Overview

Alphabet Soup wants to develop a tool that predicts whether applicants for funding will be **successful** if chosen. Using a structured dataset of metadata on previously funded organizations, the goal was to:

- Preprocess and clean the data  
- Build and evaluate a baseline deep learning model  
- Apply optimization strategies to improve model performance  
- Draw conclusions on model effectiveness and data limitations  

---

## 🧠 Technologies Used

- Python 3.8+  
- Pandas  
- scikit-learn  
- TensorFlow / Keras  
- Google Colab (for training and experimentation)

---

## 📁 Repository Structure

Deep Learning Challenge/ ├── AlphabetSoupCharity_NeuralNetwork.ipynb # Initial model and training ├── AlphabetSoupCharity_NeuralNetwork_Optimized.ipynb # Optimized model + evaluations ├── exported_models/ # All saved .h5 model files │ ├── AlphabetSoupCharity.h5 │ ├── AlphabetSoupCharity_Optimized.h5 │ ├── AlphabetSoupCharity_extraLayer.h5 │ ├── AlphabetSoupCharity_extraLayer_Optimized.h5 │ ├── AlphabetSoupCharity_tanh.h5 │ ├── AlphabetSoupCharity_tanh_Optimized.h5 └── README.md

yaml
Copy
Edit

---

## 🧼 Data Preprocessing

- **Target Variable:** `IS_SUCCESSFUL` – whether the organization used the funds effectively  
- **Feature Variables:** All columns except dropped ones  
- **Dropped Columns:** `EIN`, `NAME`, `STATUS`, `ASK_AMT`, and `SPECIAL_CONSIDERATIONS` due to imbalance or lack of value  
- **Encoding:** One-hot encoding via `pd.get_dummies()`  
- **Scaling:** Feature standardization using `StandardScaler`  

---

## 🧪 Model Architecture & Optimization

### ✅ Baseline Model

- Hidden Layers: 2 (128 → 64 nodes)  
- Activation Function: ReLU  
- Output Layer: Sigmoid  
- Optimizer: Adam  
- Loss Function: Binary Crossentropy  
- Epochs: 100  
- **Accuracy:** ~72.85%  
- **Precision:** ~72.69%  
- **Recall:** ~78.61%  

### 🔁 Optimization Attempts

Three optimization strategies were tested:

- Added a third dense layer  
- Replaced ReLU with Tanh activation  
- Dropped additional columns and reduced threshold for classification binning  

Despite these changes, the models returned consistent results. The best-performing model reached:

- **Accuracy:** 72.83%  
- **Recall:** 79.33%  

---

## 📉 Evaluation

Even after removing three low-value columns and testing different model architectures, performance stayed relatively consistent. This suggests those columns weren’t the main source of noise (though they likely contributed), since scores remained steady even after removing full columns of information.

Ultimately, the biggest limitation appears to be the dataset itself—either due to subtle, inconsistent patterns or limited predictive power across available features. Future improvements would require more robust feature engineering or rethinking the problem structure entirely.

---

## 💾 Files Included

- `AlphabetSoupCharity_NeuralNetwork.ipynb` — Initial model build  
- `AlphabetSoupCharity_NeuralNetwork_Optimized.ipynb` — Model tweaks & evaluation  
- `exported_models/` — All `.h5` model files  

---

## 📝 Summary & Next Steps

This project demonstrated the strengths and limitations of using a deep learning classifier on real-world nonprofit data. To further improve:

- Experiment with tree-based models like **Random Forests** or **XGBoost**  
- Perform advanced **feature importance analysis**  
- Apply **sampling strategies** for imbalanced classes  
- Consider enriching the dataset with additional or external features  

---

## 📌 Author

**Sean Schallberger**  
Deep Learning Challenge – University of Toronto Bootcamp