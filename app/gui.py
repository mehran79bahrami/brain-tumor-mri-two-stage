import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tkinter import Tk, Label, Button, filedialog, Canvas, NW, Scale, HORIZONTAL, Frame, LEFT, RIGHT, messagebox
from PIL import Image, ImageTk
from pathlib import Path

# --- Segmentation Helper Functions ---
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return (2. * K.sum(y_true_f * y_pred_f) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-6):
    y_true_f, y_pred_f = K.flatten(y_true), K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1 - y_true_f) * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))
    return 1.0 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


def sensitivity(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    return tp / (pos + K.epsilon())


def precision(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pred_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return tp / (pred_pos + K.epsilon())


# --- Build Classification Model ---
def build_clf_model(num_classes=4):
    base = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(128, 128, 3))
    model = tf.keras.models.Sequential([
        base, tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


# --- Settings and Model Loading ---

BASE_DIR = Path(__file__).resolve().parents[1]
SEG_PATH = BASE_DIR / "models" / "best_segmentation_model.keras"
CLF_PATH = BASE_DIR / "models" / "best_tumor_classifier.keras"

# Color mapping for each class
CLASS_COLORS = {
    "Glioma": (255, 0, 0),  # Red
    "Meningioma": (0, 255, 0),  # Green
    "Pituitary": (0, 0, 255),  # Blue (changed from yellow)
    "No Tumor": (192, 192, 192)  # Light Gray
}

print("🔄 Loading AI models...")
seg_model = tf.keras.models.load_model(SEG_PATH, custom_objects={
    'tversky_loss': tversky_loss, 'dice_coef': dice_coef,
    'sensitivity': sensitivity, 'precision': precision
})
clf_model = build_clf_model()
clf_model.load_weights(CLF_PATH)


class ModernTumorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Medical Imaging - Brain Tumor Analysis")
        self.root.geometry("1100x850")
        self.root.configure(bg="#0f111a")
        self.current_img = None
        self.tumor_details = []

        # --- Header ---
        header = Frame(root, bg="#1a1c2c", pady=10)
        header.pack(fill="x")
        Label(header, text="Brain Tumor Detection & Classification System",
              font=("Tahoma", 16, "bold"), bg="#1a1c2c", fg="#00d4ff").pack()

        # --- Toolbar ---
        toolbar = Frame(root, bg="#0f111a", pady=20)
        toolbar.pack(fill="x")

        Button(toolbar, text="📁 Load MRI Image", command=self.load,
               bg="#232946", fg="#eebbc3", font=("Tahoma", 10, "bold"),
               padx=20, borderwidth=0, cursor="hand2").pack(side=LEFT, padx=50)

        self.thresh_slider = Scale(toolbar, from_=0.01, to=0.99, resolution=0.01,
                                   orient=HORIZONTAL, label="Segmentation Threshold",
                                   bg="#0f111a", fg="#b8c1ec", length=300,
                                   highlightthickness=0, font=("Tahoma", 9))
        self.thresh_slider.set(0.5)
        self.thresh_slider.pack(side=LEFT, padx=30)

        Button(toolbar, text="🔍 Start Analysis", command=self.process,
               bg="#33d6a6", fg="#0f111a", font=("Tahoma", 10, "bold"),
               padx=30, borderwidth=0, cursor="hand2").pack(side=LEFT)

        # --- Main Display Panel (Two Panels) ---
        display_frame = Frame(root, bg="#0f111a")
        display_frame.pack(expand=True, fill="both", padx=20, pady=(0, 10))

        # Panel 1: Original Image
        self.panel_left = Frame(display_frame, bg="#1a1c2c", bd=2, relief="flat")
        self.panel_left.pack(side=LEFT, expand=True, padx=10, pady=10)
        Label(self.panel_left, text="Input MRI Image", bg="#1a1c2c", fg="#b8c1ec",
              font=("Tahoma", 10)).pack(pady=5)
        self.canvas_orig = Canvas(self.panel_left, width=400, height=400,
                                  bg="#000", highlightthickness=0)
        self.canvas_orig.pack(padx=10, pady=10)

        # Panel 2: Result
        self.panel_right = Frame(display_frame, bg="#1a1c2c", bd=2, relief="flat")
        self.panel_right.pack(side=RIGHT, expand=True, padx=10, pady=10)
        Label(self.panel_right, text="AI Output (Segmentation)",
              bg="#1a1c2c", fg="#33d6a6", font=("Tahoma", 10)).pack(pady=5)
        self.canvas_res = Canvas(self.panel_right, width=400, height=400,
                                 bg="#000", highlightthickness=0)
        self.canvas_res.pack(padx=10, pady=10)

        # --- Tumor Details Panel ---
        details_frame = Frame(root, bg="#1a1c2c", bd=2, relief="flat", height=150)
        details_frame.pack(fill="x", padx=30, pady=(0, 20))
        details_frame.pack_propagate(False)

        Label(details_frame, text="Tumor Details", bg="#1a1c2c", fg="#00d4ff",
              font=("Tahoma", 12, "bold")).pack(pady=8)

        self.details_container = Frame(details_frame, bg="#1a1c2c")
        self.details_container.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        # --- Footer Status ---
        self.status_bar = Label(root, text="Ready | Please load an MRI image",
                                bg="#1a1c2c", fg="#888", font=("Tahoma", 10), pady=10)
        self.status_bar.pack(fill="x", side="bottom")

    def load(self):
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path)
            self.current_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.display_on_canvas(cv2.resize(self.current_img, (400, 400)), self.canvas_orig)
            self.status_bar.config(text="Image loaded. Ready for analysis...", fg="#00d4ff")
            self.canvas_res.delete("all")
            for widget in self.details_container.winfo_children():
                widget.destroy()

    def process(self):
        if self.current_img is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        # 1. Segmentation
        img_256 = cv2.resize(self.current_img, (256, 256))
        raw_mask = seg_model.predict(np.expand_dims(img_256 / 255.0, 0), verbose=0)[0]

        # 2. Apply filters
        thresh = self.thresh_slider.get()
        smooth_mask = cv2.GaussianBlur(raw_mask, (5, 5), 0)
        binary_mask = (smooth_mask > thresh).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # 3. Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_img = img_256.copy()
        self.tumor_details = []

        for i, cnt in enumerate(contours, 1):
            if cv2.contourArea(cnt) < 15: continue

            # 4. Classification
            x, y, w, h = cv2.boundingRect(cnt)
            margin = 50
            crop = img_256[max(0, y - margin):min(256, y + h + margin),
            max(0, x - margin):min(256, x + w + margin)]

            if crop.size == 0:
                continue

            clf_in = np.expand_dims(cv2.resize(crop, (128, 128)) / 255.0, 0)
            predictions = clf_model.predict(clf_in, verbose=0)[0]
            class_idx = np.argmax(predictions)
            confidence = predictions[class_idx] * 100
            label = CLASS_NAMES[class_idx]

            # Get color based on predicted class
            color = CLASS_COLORS.get(label, (255, 255, 255))

            # Draw colored mask with low opacity
            mask_single = np.zeros(img_256.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_single, [cnt], -1, 255, -1)

            mask_indices = mask_single == 255
            overlay = np.zeros_like(img_256)
            overlay[mask_indices] = color
            output_img[mask_indices] = cv2.addWeighted(
                img_256[mask_indices], 0.7, overlay[mask_indices], 0.3, 0
            ).flatten().reshape(-1, 3)

            # Display tumor number to the right of bounding box
            text = f"T{i}"
            text_x = x + w + 5
            text_y = y + h // 2

            # Draw text with black border and colored text
            cv2.putText(output_img, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)  # Black border
            cv2.putText(output_img, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)  # Colored text

            # Store tumor details
            self.tumor_details.append({
                "number": i,
                "label": label,
                "confidence": confidence,
                "color": color
            })

        # Display on right panel
        self.display_on_canvas(cv2.resize(output_img, (400, 400)), self.canvas_res)

        # Update tumor details panel (all white text)
        self.update_details_panel()

        # Update status bar
        if self.tumor_details:
            self.status_bar.config(text=f"Analysis complete - {len(self.tumor_details)} tumor(s) detected",
                                   fg="#33d6a6")
        else:
            self.status_bar.config(text="Analysis complete - No tumors detected", fg="#ffaa00")

    def update_details_panel(self):
        # Clear previous details
        for widget in self.details_container.winfo_children():
            widget.destroy()

        if not self.tumor_details:
            Label(self.details_container, text="No tumors detected",
                  bg="#1a1c2c", fg="#888", font=("Tahoma", 11)).pack()
            return

        # Create a grid for tumor details
        for i, tumor in enumerate(self.tumor_details):
            frame = Frame(self.details_container, bg="#2a2c3c", bd=1, relief="flat")
            frame.pack(side=LEFT, padx=10, pady=5, ipadx=15, ipady=5)

            # Tumor number - WHITE
            Label(frame, text=f"T{tumor['number']}", bg="#2a2c3c", fg="white",
                  font=("Tahoma", 12, "bold")).pack(side=LEFT, padx=(10, 5))

            # Details - WHITE
            details_text = f"{tumor['label']} ({tumor['confidence']:.1f}%)"
            Label(frame, text=details_text, bg="#2a2c3c", fg="white",
                  font=("Tahoma", 11)).pack(side=LEFT, padx=(5, 10))

    def display_on_canvas(self, img, canvas):
        img_tk = ImageTk.PhotoImage(Image.fromarray(img))
        canvas.image = img_tk
        canvas.create_image(0, 0, anchor=NW, image=img_tk)

if __name__ == "__main__":
    root = Tk()
    app = ModernTumorApp(root)
    root.mainloop()