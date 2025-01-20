import tkinter as tk
from tkinter import messagebox
from ultralytics import YOLO
import cv2
from threading import Thread


class VegDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vegetable Cut Detector")
        self.root.geometry("400x300")  # Compact window size
        self.root.configure(bg="teal")  # Set background color

        # Load YOLO model
        self.model = YOLO(r'C:\Users\johna\PyCharmMiscProject\dataset\veg_cut_training5\weights\best.pt')

        # Variables
        self.running = False
        self.thread = None
        self.status_var = tk.StringVar(value="Status: Waiting to start detection")

        # Classes to monitor
        self.target_classes = ['Batonnet', 'LargeDice', 'FineJulienne', 'Rondelle', 'MediumDice', 'Brunoise']

        # Title Label
        title = tk.Label(root, text="Vegetable Cut Detector", bg="teal", fg="white", font=("Arial", 16, "bold"))
        title.pack(pady=20)

        # Buttons
        self.start_webcam_btn = tk.Button(root, text="Start Webcam", command=self.start_webcam, font=("Arial", 12, "bold"), bg="white", width=15)
        self.start_webcam_btn.pack(pady=10)

        self.stop_btn = tk.Button(root, text="Stop Detection", command=self.stop_detection, font=("Arial", 12, "bold"), bg="white", width=15)
        self.stop_btn.pack(pady=10)

        self.exit_btn = tk.Button(root, text="Exit", command=self.quit_app, font=("Arial", 12, "bold"), bg="white", width=15)
        self.exit_btn.pack(pady=10)

        # Status Box
        self.status_label = tk.Label(root, textvariable=self.status_var, bg="teal", fg="white", font=("Arial", 12, "italic"), wraplength=350)
        self.status_label.pack(pady=20)

    def start_webcam(self):
        if not self.running:
            self.running = True
            self.thread = Thread(target=self.detect_webcam, daemon=True)
            self.thread.start()
            self.status_var.set("Status: Detection started")

    def detect_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not access the webcam.")
            self.running = False
            self.status_var.set("Status: Error accessing webcam")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                self.status_var.set("Status: Error reading frame")
                break

            # Perform detection
            results = self.model(frame, conf=0.25)

            # Track detected classes
            detected_classes = set()

            # Process results
            for result in results:
                for box in result.boxes:
                    confidence = box.conf[0]     # Confidence score
                    class_id = int(box.cls[0])   # Class ID
                    label = self.model.names[class_id]  # Class name

                    # Draw bounding box
                    x1, y1, x2, y2 = box.xyxy[0]  # Coordinates
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Add class to detected set if it's a target class
                    if label in self.target_classes:
                        detected_classes.add(label)

            # Update status
            if detected_classes:
                self.status_var.set(f"Detected: {', '.join(detected_classes)}")
            else:
                self.status_var.set("Status: No target classes detected")

            # Display frame
            cv2.imshow('Vegetable Cut Detector', frame)

            # Stop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.running = False
        self.status_var.set("Status: Detection stopped")

    def stop_detection(self):
        if self.running:
            self.running = False
            self.status_var.set("Status: Detection stopped")

    def quit_app(self):
        self.running = False
        self.root.quit()


# Main Application
if __name__ == "__main__":
    root = tk.Tk()
    app = VegDetectorApp(root)
    root.mainloop()