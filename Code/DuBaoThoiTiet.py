import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score


label_translation = {
    "Partially cloudy": "Có mây một phần",
    "Rain, Partially cloudy": "Mưa, Có mây một phần",
    "Clear": "Trời quang",
    "Rain, Overcast": "Mưa, U ám",
    "Overcast": "U ám",
}

limits = {
    "Max Temperature": (-10, 42),
    "Min Temperature": (-10, 42),
    "Wind Speed": (0, 60),
    "Cloud Cover": (0, 100),
    "Relative Humidity": (0, 100),
}

def load_icon(name):
    icons = {
        "Clear": "sunny_icon.png",
        "Partially cloudy": "partly_cloudy_icon.png",
        "Rain, Partially cloudy": "rainy_partly_cloudy_icon.png",
        "Rain, Overcast": "rainy_overcast_icon.png",
        "Overcast": "overcast.png",
    }

    icon_file = icons.get(name, "default_icon.png")
    path = os.path.join(os.path.dirname(__file__), icon_file)

    try:
        img = Image.open(path).resize((70, 70))
        return ImageTk.PhotoImage(img)
    except:
        return None


# -----------------------------
# LOAD VÀ TRAIN MODEL
# -----------------------------

def train_model():
    data_path = os.path.join(os.path.dirname(__file__), "ThoiTiet_dulieu.csv")
    data = pd.read_csv(data_path)

    target = "Conditions"
    X = data.drop(target, axis=1)
    y = data[target]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", transformer, ["Max Temperature", "Min Temperature", "Wind Speed", "Cloud Cover", "Relative Humidity"])
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier())
    ])

    model.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(x_test))

    return model, accuracy


# -----------------------------
# GIAO DIỆN TKINTER
# -----------------------------

def create_gui(model, accuracy):

    root = tk.Tk()
    root.title("Dự Báo Thời Tiết - sklearn version")

    window_width = 550
    window_height = 600
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{window_width}x{window_height}+{sw//2 - window_width//2}+{sh//2 - window_height//2}")

    labels = ["Max Temperature", "Min Temperature", "Wind Speed", "Cloud Cover", "Relative Humidity"]
    units  = ["℃", "℃", "km/h", "%", "%"]

    entries = {}
    error_labels = {}

    frame = tk.Frame(root)
    frame.pack(pady=10)

    for i, (lb, unit) in enumerate(zip(labels, units)):
        tk.Label(frame, text=f"{lb} ({unit})").grid(row=i*2, column=0, sticky="w")

        entry = tk.Entry(frame, font=("Arial", 12))
        entry.grid(row=i*2, column=1, pady=5)
        entries[lb] = entry

        err = tk.Label(frame, text="", fg="red")
        err.grid(row=i*2+1, column=1, sticky="w")
        error_labels[lb] = err

    result_label = tk.Label(root, text="Dự đoán: Chưa có dự đoán.", font=("Arial", 13))
    result_label.pack(pady=10)

    icon_label = tk.Label(root)
    icon_label.pack(pady=10)

    accuracy_label = tk.Label(root, text=f"Tỷ lệ chính xác mô hình: {accuracy*100:.2f}%")
    accuracy_label.pack(pady=10)

    # -----------------------------
    # XỬ LÝ KHI NHẤN NÚT DỰ ĐOÁN
    # -----------------------------

    def submit():
        user_input = {}

        # Validate input
        for key, entry in entries.items():
            value = entry.get()
            try:
                num = float(value)
                min_v, max_v = limits[key]

                if not (min_v <= num <= max_v):
                    error_labels[key].config(text=f"Giá trị phải trong khoảng {min_v} đến {max_v}")
                    return
                else:
                    error_labels[key].config(text="")

                user_input[key] = num

            except:
                error_labels[key].config(text="Giá trị không hợp lệ!")
                return

        df_input = pd.DataFrame([user_input])
        prediction = model.predict(df_input)[0]
        translated = label_translation.get(prediction, prediction)

        # show text
        result_label.config(text=f"Dự đoán: {translated}", font=("Arial", 14, "bold"))

        # show icon
        icon = load_icon(prediction)
        if icon:
            icon_label.config(image=icon)
            icon_label.image = icon

    tk.Button(root, text="Dự đoán", command=submit,
              bg="green", fg="white", font=("Arial", 12)).pack(pady=10)

    root.mainloop()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    model, acc = train_model()
    create_gui(model, acc)
