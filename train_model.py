import os
import time
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from face_mask_rnn_model import FaceMaskRNNModel

def load_dataset(data_dir="data", max_per_class=1250):
    wm_dir = os.path.join(data_dir, "with_mask")
    wom_dir = os.path.join(data_dir, "without_mask")
    
    wm_files = [os.path.join(wm_dir, f) for f in os.listdir(wm_dir) if f.lower().endswith(('.jpg', '.png'))][:max_per_class]
    wom_files = [os.path.join(wom_dir, f) for f in os.listdir(wom_dir) if f.lower().endswith(('.jpg', '.png'))][:max_per_class]
    
    print(f"Loading {len(wm_files)} 'with_mask' and {len(wom_files)} 'without_mask' images...")
    X_list, y_list = [], []
    
    for p in wm_files:
        img = cv2.imread(p)
        if img is not None:
            img = cv2.resize(img, (32, 32))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_norm = (img_rgb.astype(np.float32) / 127.5) - 1.0
            X_list.append(img_norm.reshape(32, 96))
            y_list.append(1.0)
            
    for p in wom_files:
        img = cv2.imread(p)
        if img is not None:
            img = cv2.resize(img, (32, 32))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_norm = (img_rgb.astype(np.float32) / 127.5) - 1.0
            X_list.append(img_norm.reshape(32, 96))
            y_list.append(0.0)
            
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    return X, y

def main():
    print("=" * 65)
    print(" TRAINING HIGH-ACCURACY FACE MASK DETECTION RNN")
    print("=" * 65)
    
    X, y = load_dataset("data", max_per_class=1250)
    print(f"Total samples: {len(X)}, Shape: {X.shape}")
    
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    model = FaceMaskRNNModel(input_dim=96, hidden_dim=64, timesteps=32, learning_rate=0.003)
    
    print("\nStarting Training (25 Epochs)...")
    t0 = time.time()
    history = model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=25,
        batch_size=64,
        verbose=True
    )
    print(f"Training completed in {time.time() - t0:.2f}s.")
    
    # Evaluate on test set
    _, _, y_test_probs = model.forward(X_test)
    preds = (y_test_probs >= 0.5).astype(int).flatten()
    actual = y_test.astype(int).flatten()
    
    acc = float(np.mean(preds == actual))
    tp = int(np.sum((preds == 1) & (actual == 1)))
    tn = int(np.sum((preds == 0) & (actual == 0)))
    fp = int(np.sum((preds == 1) & (actual == 0)))
    fn = int(np.sum((preds == 0) & (actual == 1)))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "=" * 50)
    print(" TEST SET PERFORMANCE METRICS")
    print("=" * 50)
    print(f"Accuracy:   {acc * 100:.2f}%")
    print(f"Precision:  {precision * 100:.2f}%")
    print(f"Recall:     {recall * 100:.2f}%")
    print(f"F1-Score:   {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"                 Predicted No Mask   Predicted Mask")
    print(f"Actual No Mask         {tn:4d}             {fp:4d}")
    print(f"Actual Mask            {fn:4d}             {tp:4d}")
    
    # Save to pickle file
    model.save("face_mask_rnn_model.pkl")
    
    # Plot curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='#2563eb', lw=2)
    plt.plot(history['val_loss'], label='Val Loss', color='#dc2626', lw=2, ls='--')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross-Entropy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot([a * 100 for a in history['train_acc']], label='Train Acc', color='#2563eb', lw=2)
    plt.plot([a * 100 for a in history['val_acc']], label='Val Acc', color='#16a34a', lw=2, ls='--')
    plt.title('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("\nSaved training curves to 'training_curves.png'.")
    
    # Verify predictions on sample mask images
    print("\n" + "=" * 50)
    print(" VERIFYING PREDICTIONS ON KNOWN SAMPLES")
    print("=" * 50)
    samples_mask = [f"data/with_mask/{f}" for f in os.listdir("data/with_mask")[:5]]
    samples_nomask = [f"data/without_mask/{f}" for f in os.listdir("data/without_mask")[:5]]
    
    for s in samples_mask:
        res = model.predict(s)
        print(f"MASK:   {os.path.basename(s):20s} -> {res['label']} ({res['confidence']*100:.1f}%)")
        
    for s in samples_nomask:
        res = model.predict(s)
        print(f"NOMASK: {os.path.basename(s):20s} -> {res['label']} ({res['confidence']*100:.1f}%)")

if __name__ == "__main__":
    main()
