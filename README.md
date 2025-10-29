
# Brain Tumor Detection from MRI using Federated Learning

Privacy-preserving brain-tumor classification on MRI images using a simple CNN and a **federated learning (FL)** workflow that simulates two clients (Raspberry Pi 1 & 2). The global model is updated by averaging client weights—improving generalization on unseen data without centralizing patient data. 

---

## ✨ Highlights

* **Federated training loop:** main model → distribute weights → train clients → average weights → update global model. 
* **Lightweight CNN:** 3×Conv + MaxPool backbone (ReLU), dense head; binary cross-entropy, Adam optimizer. 
* **Edge focus:** TensorFlow/Keras workflow with export to **TFLite** for embedded devices. 
* **Results (indicative):** Main training acc ~**88%**; Client eval on unseen data up to **+9%** after FL weight averaging (RPi1 ~89%, RPi2 ~80%). 

---

## 🧠 Problem & Approach

Centralized medical image sharing is restricted; FL keeps data local and only shares **model updates**. We train a baseline CNN on a main split, push weights to two “clients” (Raspberry Pi 1 & 2), train locally, **average** client weights, and update the global model. Evaluate on **unseen** data to assess generalization. 

---

## 📚 Dataset

* **Brain MRI** (binary labels: *tumor yes/no*), **253** images (≈155 yes / 98 no).
* Splits: ~60% main model; ~20% RPi1; ~20% RPi2; labels saved to CSV (e.g., `main_data.csv`, `rasp1_data.csv`, `rasp2_data.csv`).
* **Augmentation:** horizontal flips, random rotations (~20%) to balance/regularize.
* GANs were explored but not used due to very small dataset; classic augmentation performed better here. 

> ⚠️ Please verify license/usage permissions for the original dataset source before redistribution.

---

## 🏗️ Project Structure (suggested)

```
repo-root/
├─ data/
│  ├─ main/         # images for main model
│  ├─ rasp1/        # images for client 1
│  └─ rasp2/        # images for client 2
├─ labels/
│  ├─ main_data.csv
│  ├─ rasp1_data.csv
│  └─ rasp2_data.csv
├─ src/
│  ├─ models.py           # get_model(), simple vs extra_dense_CNN
│  ├─ data.py             # loaders/augmentations
│  ├─ train_main.py       # trains baseline (global init)
│  ├─ train_client.py     # trains a client from given init weights
│  ├─ federated.py        # weight avg & global update
│  ├─ evaluate.py         # eval on unseen sets (per client)
│  └─ export_tflite.py    # TFLite export for edge
├─ notebooks/             # (optional) experiment notebooks
├─ results/               # metrics, plots
├─ requirements.txt
└─ README.md
```

---

## ⚙️ Setup

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt (example)**

```
tensorflow>=2.12
numpy
pandas
scikit-learn
matplotlib
```

---

## 🚀 Usage

### 1) Prepare data & labels

Place images under `data/` as shown above and generate CSV label files (`labels/*.csv`) with file path and binary label (0/1). 

### 2) Train the main (global-init) model

```bash
python src/train_main.py \
  --data_dir data/main \
  --labels labels/main_data.csv \
  --epochs 10 \
  --out results/main
```

* Saves baseline weights: `results/main/main_weights.h5`. 

### 3) Train client models (local updates)

```bash
# Client 1
python src/train_client.py \
  --init_weights results/main/main_weights.h5 \
  --data_dir data/rasp1 \
  --labels labels/rasp1_data.csv \
  --model_variant extra_dense_CNN \
  --epochs 10 \
  --out results/rasp1

# Client 2
python src/train_client.py \
  --init_weights results/main/main_weights.h5 \
  --data_dir data/rasp2 \
  --labels labels/rasp2_data.csv \
  --model_variant extra_dense_CNN \
  --epochs 10 \
  --out results/rasp2
```

* Produces `rasp1_weights.h5`, `rasp2_weights.h5`. 

### 4) Federated averaging & global update

```bash
python src/federated.py \
  --client_weights results/rasp1/rasp1_weights.h5 results/rasp2/rasp2_weights.h5 \
  --global_init results/main/main_weights.h5 \
  --out results/global/global_weights.h5
```

* Averages client weights and updates the global model. 

### 5) Evaluate on unseen data (per client)

```bash
python src/evaluate.py \
  --weights results/global/global_weights.h5 \
  --data_dir data/rasp1_unseen \
  --labels labels/rasp1_unseen.csv \
  --out results/eval_rasp1

python src/evaluate.py \
  --weights results/global/global_weights.h5 \
  --data_dir data/rasp2_unseen \
  --labels labels/rasp2_unseen.csv \
  --out results/eval_rasp2
```

* Expect improved generalization for clients after FL (e.g., ≈+9% in reported experiments). 

### 6) Export to TFLite (optional)

```bash
python src/export_tflite.py \
  --weights results/global/global_weights.h5 \
  --out results/global/model.tflite
```

* For embedded/edge deployment. 

---

## 📈 Indicative Results

* **Main model (train acc):** ~**88%**
* **Client eval on unseen data:**

  * RPi1: ~**89%** after FL update vs ~80% before
  * RPi2: ~**80%** (varies by split/seed)
* Small dataset → overfitting risks; keep epochs modest and use augmentation. 

---

## 🧩 Model Details

* **Backbone:** 3 conv blocks (8→16→32 filters), MaxPool, Flatten, Dense(32), Dense(1).
* **Loss/Opt:** BinaryCrossentropy + Adam.
* **Client variant:** adds extra conv/dense layers; earlier layers frozen for transfer learning. 

---

## ⚠️ Limitations & Notes

* **Data scarcity/class imbalance** can hurt stability; consider stronger augmentation or careful re-splits.
* Client data skew (e.g., more “no-tumor” samples on a client) may bias global updates.
* Some devices may lack compute for on-device training. 

---

## 🛣️ Roadmap

* [ ] Add stratified splits and reproducible seeds
* [ ] Track metrics with TensorBoard and save ROC/PR curves
* [ ] Experiment with **FedAvg** variants and differential privacy
* [ ] Evaluate lightweight pretrained backbones (e.g., MobileNet-V2)
* [ ] Improve TFLite quantization (int8) and measure latency on edge 

---

## 📜 Citation / Reference

If you use this repository or build upon the reported setup/results, please cite or reference the accompanying write-up/presentation for methodology and metrics. 

---

## 📝 License

Add your preferred license (e.g., MIT) here. Make sure the dataset’s license permits your intended use/distribution.

---

## 🙏 Acknowledgments

Thanks to the collaborators and prior work on federated learning and medical imaging referenced in the project write-up. 


