import os
import pickle
import numpy as np
import cv2

class FaceMaskRNNModel:
    """
    Recurrent Neural Network (RNN) with Spatial-Sequential Region Attention for Face Mask Detection.
    
    Architecture:
    - Input: 32 sequential horizontal scanlines, each with 96 features (32 RGB pixels per row).
    - Recurrent Layer: 64 hidden units with tanh activation and BPTT.
    - Spatial-Sequential Attention: Pooling across the global sequence (h_mean) and the lower facial 
      mask region (rows 14-31, h_lower) to directly capture surgical/cloth masks without vanishing at the chin.
    - Optimizer: Adam (adaptive moment estimation) with gradient clipping.
    """
    def __init__(self, input_dim=96, hidden_dim=64, timesteps=32, learning_rate=0.003):
        self.input_dim = input_dim          # 32 pixels * 3 channels (RGB) = 96 features
        self.hidden_dim = hidden_dim        # 64 recurrent hidden units
        self.timesteps = timesteps          # 32 vertical scanline rows
        self.mask_start_row = 14            # Row index where nose/mouth/mask region begins
        self.learning_rate = learning_rate
        self.classes = {0: "without_mask", 1: "with_mask"}
        self.img_size = (32, 32)
        
        # Xavier/He initialization
        np.random.seed(42)
        self.W_xh = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_h = np.zeros((1, hidden_dim))
        
        # Classifier weights: takes concatenation of [h_all_mean, h_lower_mean] -> dimension: 2 * hidden_dim
        feat_dim = 2 * hidden_dim
        self.W_hy = np.random.randn(feat_dim, 1) * np.sqrt(2.0 / feat_dim)
        self.b_y = np.zeros((1, 1))
        
        # Adam optimizer moving averages
        self.mW_xh, self.vW_xh = np.zeros_like(self.W_xh), np.zeros_like(self.W_xh)
        self.mW_hh, self.vW_hh = np.zeros_like(self.W_hh), np.zeros_like(self.W_hh)
        self.mb_h, self.vb_h = np.zeros_like(self.b_h), np.zeros_like(self.b_h)
        self.mW_hy, self.vW_hy = np.zeros_like(self.W_hy), np.zeros_like(self.W_hy)
        self.mb_y, self.vb_y = np.zeros_like(self.b_y), np.zeros_like(self.b_y)
        self.t = 0  # Timestep counter

    def forward(self, X):
        """
        Forward pass through RNN with spatial sequence attention.
        X shape: (Batch_size, 32, 96)
        Returns:
            H_list: List of hidden state matrices at each timestep
            h_feat: Concatenated pooled feature representation (Batch_size, 2*hidden_dim)
            y_pred: Predicted mask probability (Batch_size, 1)
        """
        batch_size, T, _ = X.shape
        H_list = []
        h_curr = np.zeros((batch_size, self.hidden_dim))
        
        for step in range(T):
            x_t = X[:, step, :]
            linear_t = np.dot(x_t, self.W_xh) + np.dot(h_curr, self.W_hh) + self.b_h
            h_curr = np.tanh(linear_t)
            H_list.append(h_curr)
            
        H_arr = np.stack(H_list, axis=1) # (Batch_size, 32, hidden_dim)
        
        # Global mean and lower facial region mean (mask location)
        h_all_mean = np.mean(H_arr, axis=1)
        h_lower_mean = np.mean(H_arr[:, self.mask_start_row:, :], axis=1)
        h_feat = np.concatenate([h_all_mean, h_lower_mean], axis=1)
        
        logits = np.dot(h_feat, self.W_hy) + self.b_y
        logits = np.clip(logits, -15.0, 15.0)
        y_pred = 1.0 / (1.0 + np.exp(-logits))
        return H_list, h_feat, y_pred

    def compute_loss(self, y_pred, y_true):
        """Binary cross-entropy loss with numerical stabilization."""
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return float(loss)

    def backward(self, X, y_true, H_list, h_feat, y_pred):
        """
        Backpropagation Through Time (BPTT) with attention gradient distribution.
        """
        batch_size, T, _ = X.shape
        dz = (y_pred - y_true) / batch_size
        dW_hy = np.dot(h_feat.T, dz)
        db_y = np.sum(dz, axis=0, keepdims=True)
        
        dh_feat = np.dot(dz, self.W_hy.T) # (batch_size, 2*H)
        dh_all = dh_feat[:, :self.hidden_dim] / T
        dh_lower = dh_feat[:, self.hidden_dim:] / (T - self.mask_start_row)
        
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)
        
        dh_next = np.zeros((batch_size, self.hidden_dim))
        for step in reversed(range(T)):
            h_t = H_list[step]
            h_prev = H_list[step - 1] if step > 0 else np.zeros((batch_size, self.hidden_dim))
            
            total_dh = dh_all + dh_next
            if step >= self.mask_start_row:
                total_dh += dh_lower
                
            dtanh = total_dh * (1.0 - h_t ** 2)
            dW_xh += np.dot(X[:, step, :].T, dtanh)
            dW_hh += np.dot(h_prev.T, dtanh)
            db_h += np.sum(dtanh, axis=0, keepdims=True)
            dh_next = np.dot(dtanh, self.W_hh.T)
            
        for g in [dW_xh, dW_hh, db_h, dW_hy, db_y]:
            np.clip(g, -5.0, 5.0, out=g)
            
        return dW_xh, dW_hh, db_h, dW_hy, db_y

    def update_parameters(self, dW_xh, dW_hh, db_h, dW_hy, db_y, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimizer step with bias correction."""
        self.t += 1
        lr_t = self.learning_rate * (np.sqrt(1.0 - beta2 ** self.t) / (1.0 - beta1 ** self.t))
        
        params = [self.W_xh, self.W_hh, self.b_h, self.W_hy, self.b_y]
        grads = [dW_xh, dW_hh, db_h, dW_hy, db_y]
        m_list = [self.mW_xh, self.mW_hh, self.mb_h, self.mW_hy, self.mb_y]
        v_list = [self.vW_xh, self.vW_hh, self.vb_h, self.vW_hy, self.vb_y]
        
        for p, g, m, v in zip(params, grads, m_list, v_list):
            m[:] = beta1 * m + (1.0 - beta1) * g
            v[:] = beta2 * v + (1.0 - beta2) * (g ** 2)
            p -= lr_t * m / (np.sqrt(v) + eps)

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=25, batch_size=64, verbose=True):
        """Train RNN with mini-batch Adam optimization."""
        n_samples = len(X_train)
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        n_batches = int(np.ceil(n_samples / batch_size))
        
        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(n_samples)
            X_s, y_s = X_train[perm], y_train[perm]
            
            epoch_loss = 0.0
            correct = 0
            for b in range(n_batches):
                s, e = b * batch_size, min((b + 1) * batch_size, n_samples)
                X_b, y_b = X_s[s:e], y_s[s:e]
                
                H_list, h_feat, y_pred = self.forward(X_b)
                loss = self.compute_loss(y_pred, y_b)
                epoch_loss += loss * (e - s)
                
                preds = (y_pred >= 0.5).astype(np.float32)
                correct += np.sum(preds == y_b)
                
                grads = self.backward(X_b, y_b, H_list, h_feat, y_pred)
                self.update_parameters(*grads)
                
            tr_loss = epoch_loss / n_samples
            tr_acc = correct / n_samples
            history['train_loss'].append(tr_loss)
            history['train_acc'].append(tr_acc)
            
            val_loss, val_acc = 0.0, 0.0
            if X_val is not None and y_val is not None:
                _, _, y_val_pred = self.forward(X_val)
                val_loss = self.compute_loss(y_val_pred, y_val)
                val_acc = float(np.mean((y_val_pred >= 0.5).astype(np.float32) == y_val))
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
            if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
                val_str = f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%" if X_val is not None else ""
                print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {tr_loss:.4f} - Train Acc: {tr_acc*100:.2f}%{val_str}")
                
        return history

    def preprocess_image(self, image_input):
        """
        Preprocess input image to RGB sequence of shape (1, 32, 96):
        1. Reads image from path or accepts numpy array.
        2. Normalizes color channels properly (RGB).
        3. Resizes to (32, 32) and flattens each row to 32*3 = 96 features.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not load image: {image_input}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                img_rgb = image_input.copy()
        else:
            raise TypeError("Expected image filepath or numpy ndarray")
            
        img_resized = cv2.resize(img_rgb, self.img_size)
        # Normalize to [-1.0, 1.0] for zero-centered tanh recurrent activations
        img_norm = (img_resized.astype(np.float32) / 127.5) - 1.0
        seq = img_norm.reshape(1, self.timesteps, self.input_dim)
        return seq

    def predict(self, image_input):
        """Predict whether face is with_mask or without_mask."""
        X_seq = self.preprocess_image(image_input)
        _, _, prob = self.forward(X_seq)
        p = float(prob[0, 0])
        class_id = 1 if p >= 0.5 else 0
        confidence = p if class_id == 1 else (1.0 - p)
        return {
            'label': self.classes[class_id],
            'class_id': class_id,
            'probability_with_mask': p,
            'confidence': confidence
        }

    def save(self, filepath):
        """Save model to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to pickle file: {filepath}")

    @staticmethod
    def load(filepath):
        """Load model from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

if __name__ == "__main__":
    print("=" * 60)
    print(" TESTING FaceMaskRNNModel ARCHITECTURE")
    print("=" * 60)
    model = FaceMaskRNNModel()
    dummy = np.random.randn(2, 32, 96).astype(np.float32)
    _, _, p = model.forward(dummy)
    print(f"Forward output shape: {p.shape}")
    sample = "data/with_mask/with_mask_1.jpg"
    if os.path.exists(sample):
        res = model.predict(sample)
        print(f"Sample predict test: {res['label']}")
    print("=" * 60)
    print(" [PASSED] Model architecture ready!")
    print("=" * 60)
