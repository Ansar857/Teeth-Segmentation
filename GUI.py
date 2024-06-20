import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
from ultralytics import YOLO
import pandas as pd
import os
import cv2
import numpy as np
import threading
import matplotlib.pyplot as plt
from skimage.measure import regionprops

class ToothSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tooth Segmentation App")

        # Get screen width and height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Set the size of the main window to fill the entire screen
        self.root.geometry(f"{screen_width}x{screen_height}")

        upload_button = tk.Button(root, bg="#ffd700", text='Upload Image', command=self.upload_image, width=15, height=1, relief=tk.SOLID, font=('arial', 18))
        upload_button.place(x=100, y=100)

        clear_button = tk.Button(root, bg="#ffd700", text='Clear All', command=self.clear_all, width=15, height=1, relief=tk.SOLID, font=('arial', 18))
        clear_button.place(x=100, y=200)

        self.model = YOLO("best.pt")
        self.labels = []
        self.images = []

        # Initialize loader window
        self.loader_window = tk.Toplevel(self.root)
        self.loader_window.title("Loading...")
        self.loader_window.withdraw()  # Hide loader initially
        self.loader_label = tk.Label(self.loader_window, text="Processing...", font=("Arial", 12))
        self.loader_label.pack(pady=20)

    def show_loader(self):
        # Show loader in the center of the main window
        self.loader_window.deiconify()
        x = (self.root.winfo_screenwidth() - self.loader_window.winfo_reqwidth()) / 2
        y = (self.root.winfo_screenheight() - self.loader_window.winfo_reqheight()) / 2
        self.loader_window.geometry("+%d+%d" % (x, y))

    def hide_loader(self):
        # Hide loader
        self.loader_window.withdraw()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.clear_all()  # Clear previous labels and images before uploading new image
            threading.Thread(target=self.predict_and_display, args=(file_path,)).start()

    def clear_all(self):
        for label in self.labels:
            label.destroy()
        self.labels = []

        for image_label, photo in self.images:
            image_label.destroy()
        self.images = []

    def predict_and_display(self, image_path):
        try:
            self.show_loader()  # Show loader while processing

            results = self.model.predict(image_path, conf=0.7)
            results = results[0]

            extracted_masks = results.masks.data
            masks_array = extracted_masks.cpu().numpy()
            class_names = results.names.values()
            detected_boxes = results.boxes.data
            class_labels = detected_boxes[:, -1].int().tolist()
            masks_by_class = {name: [] for name in results.names.values()}

            for mask, class_id in zip(extracted_masks, class_labels):
                class_name = results.names[class_id]  # Map class ID to class name
                masks_by_class[class_name].append(mask.cpu().numpy())

            for class_name, masks in masks_by_class.items():
                print("Number of Solid Teeths:", len(masks))

                # DISPLAYING NUMBER OF TEETHS

                result_label = tk.Label(text="Number of Solid Teeths: {}".format(len(masks)), relief=tk.SOLID, bg="#ffcccb", width=43, height=1, font=('arial', 11))
                result_label.place(x=600, y=10)
                self.labels.append(result_label)

            orig_img = results.orig_img
            teeth_mask = masks_by_class['T']
            teeth_masks_sorted = sorted(teeth_mask, key=lambda x: np.count_nonzero(x), reverse=True)

            overlay_img = orig_img.copy()
            num_teeth_to_display = min(2, len(teeth_masks_sorted))  # Display up to 2 teeth
            for i in range(num_teeth_to_display):
                overlay_img[teeth_masks_sorted[i] != 0] = [255, 255, 0]  # Set mask region to red (BGR format)
            cv2.imwrite("Predicted.png", overlay_img)

            props_list = []
            y_position = 120
            for class_name, masks in masks_by_class.items():
                for mask in masks:
                    mask = mask.astype(int)
                    props = regionprops(mask)
                    for prop in props:
                        area = prop.area
                        perimeter = prop.perimeter
                        props_list.append({'Class Name': class_name, 'Area': area, 'Perimeter': perimeter})
                        label = tk.Label(text=f"A : {area} \n P : {perimeter}", bg="#ffcccb", relief=tk.SOLID, height=2,width=20, font=('arial', 11))
                        label.pack(side=tk.LEFT, padx=10, pady=5)
                        y_position += 100
                        self.labels.append(label)

            props_df = pd.DataFrame(props_list)

            for i, tooth_mask in enumerate(masks_by_class['T']):
                output_dir = 'segmented_teeth'
                os.makedirs(output_dir, exist_ok=True)
                segmented_tooth = orig_img.copy()
                segmented_tooth[tooth_mask == 0] = 0  # Set pixels outside the tooth mask to zero
                transparent_tooth = np.zeros((segmented_tooth.shape[0], segmented_tooth.shape[1], 4), dtype=np.uint8)
                transparent_tooth[:, :, :3] = segmented_tooth
                transparent_tooth[:, :, 3] = tooth_mask * 255  # Scale mask values to 0-255
                tooth_filename = os.path.join(output_dir, f'tooth_{i}.png')
                cv2.imwrite(tooth_filename, transparent_tooth)
                self.images.append(self.load_image(tooth_filename))

            self.load_images()

            # Predicted Image

            image = Image.open("Predicted.png")
            image = image.resize((400, 300))
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(self.root, image=photo)
            label.image = photo
            label.place(x=600, y=35)
            self.images.append((label, photo))

            self.hide_loader()  # Hide loader when processing is done

        except Exception as e:
            self.hide_loader()  # Hide loader if an error occurs
            messagebox.showerror("Error", str(e))

    def load_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((180, 150))
        photo = ImageTk.PhotoImage(image)
        return (tk.Label(self.root, image=photo, relief=tk.SOLID, bg='red'), photo)

    def load_images(self):
        frame = tk.Frame(self.root)
        frame.place(x=10, y=500)
        x_position = 0
        for image_label, _ in self.images:
            image_label.place(x=10+x_position , y=500)
            x_position +=206

if __name__ == "__main__":
    root = tk.Tk()
    app = ToothSegmentationApp(root)
    root.mainloop()
