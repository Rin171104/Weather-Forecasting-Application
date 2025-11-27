# Thư viện cần thiết
import csv  # Thư viện để đọc/ghi tệp CSV
import numpy as np  # Thư viện để tính toán số học và ma trận
import pandas as pd  # Thư viện để xử lý dữ liệu dạng bảng
import os  # Thư viện để làm việc với đường dẫn và tệp tin
from collections import Counter  # Thư viện để đếm tần suất các giá trị
import tkinter as tk  # Thư viện GUI để tạo giao diện người dùng
from PIL import Image, ImageTk  # Thư viện để xử lý và hiển thị hình ảnh

# Dictionary để dịch nhãn từ tiếng Anh sang tiếng Việt
label_translation = {
    "Partially cloudy": "Có mây một phần",  # Dịch nhãn có mây một phần
    "Rain, Partially cloudy": "Mưa, Có mây một phần",  # Dịch nhãn mưa và có mây một phần
    "Clear": "Trời quang",  # Dịch nhãn trời quang
    "Rain, Overcast": "Mưa, U ám",  # Dịch nhãn mưa và u ám
    "Overcast": "U ám",  # Dịch nhãn u ám
}

# Hàm để tạo đường dẫn tương đối
def get_relative_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)


# Hàm làm sạch và kiểm tra dữ liệu đầu vào
def clean_input_data(user_data, error_labels):
    # Giới hạn cho từng trường
    limits = {
        "Max Temperature": (-10, 42),
        "Min Temperature": (-10, 42),
        "Wind Speed": (0, 60),
        "Cloud Cover": (0, 100),
        "Relative Humidity": (0, 100),
    }

    cleaned_data = {}  # Dữ liệu sau khi làm sạch
    error_found = False  # Biến kiểm tra nếu có lỗi
    for key, value in user_data.items():
        try:
            num_value = float(value)  # Chuyển đổi giá trị người dùng nhập thành số
            min_val, max_val = limits[key]  # Lấy giới hạn của trường dữ liệu
            # Kiểm tra giá trị có trong giới hạn cho phép không
            if min_val <= num_value <= max_val:
                cleaned_data[key] = num_value
                error_labels[key].config(text="")  # Xóa thông báo lỗi nếu hợp lệ
            else:
                error_labels[key].config(
                    text=f"{key} vượt quá giới hạn cho phép! Mời nhập lại"
                )  # Thông báo lỗi nếu giá trị vượt quá giới hạn
                error_found = True
        except ValueError:  # Xử lý khi giá trị không hợp lệ
            error_labels[key].config(
                text=f"{key} không hợp lệ! Mời nhập lại."
            )  # Thông báo lỗi nếu không phải số
            error_found = True

    return cleaned_data, error_found


# Load dữ liệu từ CSV và phân chia thành tập train và test
def loadData(filename="ThoiTiet_dulieu.csv"):
    path = get_relative_path(filename)  # Lấy đường dẫn đến tệp CSV
    with open(path) as f:
        data = np.array(
            list(csv.reader(f))[1:]
        )  # Đọc dữ liệu từ tệp và bỏ qua dòng tiêu đề
    np.random.shuffle(data)  # Xáo trộn dữ liệu ngẫu nhiên
    return data[:362], data[362:]  # Trả về tập huấn luyện và kiểm tra


# Tính khoảng cách giữa hai điểm
def calcDistance(pointA, pointB):
    return np.linalg.norm(
        pointA[:5].astype(float) - pointB[:5].astype(float)
    )  # Tính khoảng cách Euclid


# Thuật toán k-NN
def kNearestNeighbor(trainSet, point, k=7):
    distances = [
        {"label": item[-1], "value": calcDistance(item, point)} for item in trainSet
    ]  # Tính khoảng cách và gán nhãn
    return [
        item["label"] for item in sorted(distances, key=lambda x: x["value"])[:k]
    ]  # Trả về k nhãn gần nhất


# Tìm nhãn xuất hiện nhiều nhất
def findMostOccur(labels):
    return Counter(labels).most_common(1)[0][0]  # Trả về nhãn xuất hiện nhiều nhất


# Load dữ liệu dự đoán từ file CSV
def loadDataInput(filename):
    path = get_relative_path(filename)
    with open(path, "r") as f:
        return np.array(list(csv.reader(f))[1:2])  # Đọc dữ liệu cần dự đoán từ tệp


# Hàm tính tỷ lệ chính xác (accuracy)
def calculate_accuracy(testSet, trainSet, k=5):
    correct_predictions = 0
    for item in testSet:
        # Dự đoán nhãn cho từng mẫu trong testSet
        knn = kNearestNeighbor(trainSet, item, k)
        predicted_label = findMostOccur(knn)  # Lấy nhãn dự đoán
        actual_label = item[-1]  # Lấy nhãn thực tế
        if predicted_label == actual_label:  # Kiểm tra nếu dự đoán đúng
            correct_predictions += 1
    accuracy = correct_predictions / len(testSet) * 100  # Tính tỷ lệ chính xác
    return accuracy


# Hàm để tải icon dựa trên dự đoán
def load_icon(prediction):
    icons = {
        "Clear": "sunny_icon.png",  # Icon cho trời quang
        "Partially cloudy": "partly_cloudy_icon.png",  # Icon cho có mây một phần
        "Rain, Partially cloudy": "rainy_partly_cloudy_icon.png",  # Icon cho mưa có mây một phần
        "Rain, Overcast": "rainy_overcast_icon.png",  # Icon cho mưa u ám
        "Overcast": "overcast.png",  # Icon cho u ám
    }

    icon_path = get_relative_path(
        icons.get(prediction, "default_icon.png")
    )  # Lấy đường dẫn đến icon dựa trên dự đoán
    icon_image = Image.open(icon_path)  # Mở file ảnh
    icon_image = icon_image.resize((70, 70))  # Điều chỉnh kích thước cho vừa với GUI
    return ImageTk.PhotoImage(icon_image)  # Trả về icon đã chuẩn bị


# Hàm tạo giao diện người dùng với tkinter
def create_gui():
    # Create the main window
    root = tk.Tk()  # Tạo cửa sổ chính
    root.title("Dự Báo Thời Tiết")  # Tiêu đề của cửa sổ

    # Lấy kích thước màn hình để căn giữa cửa sổ
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Cài đặt kích thước cửa sổ và vị trí để căn giữa
    window_width = 600
    window_height = 600
    position_top = int(screen_height / 2 - window_height / 2)
    position_left = int(screen_width / 2 - window_width / 2)
    root.geometry(f"{window_width}x{window_height}+{position_left}+{position_top}")

    # Tạo các nhãn và ô nhập liệu cho dữ liệu người dùng
    labels = [
        "Max Temperature",
        "Min Temperature",
        "Wind Speed",
        "Cloud Cover",
        "Relative Humidity",
    ]
    units = ["(℃)", "(℃)", "(km/h)", "(%)", "(%)"]  # Mảng chứa các ký hiệu đơn vị
    entries = {}  # Dictionary lưu trữ các ô nhập liệu
    error_labels = {}  # Dictionary lưu trữ các nhãn thông báo lỗi

    # Tạo một frame để chứa các nhãn và ô nhập liệu
    form_frame = tk.Frame(root)  # Tạo frame con trong cửa sổ chính
    form_frame.pack(padx=10, pady=10)  # Thêm khoảng cách xung quanh frame

    # Duyệt qua danh sách nhãn và đơn vị
    for i, (label, unit) in enumerate(zip(labels, units)):
        # Tạo nhãn cho mỗi trường dữ liệu, bao gồm cả đơn vị
        tk.Label(form_frame, text=f"{label} {unit}").grid(
            row=i * 2, column=0, sticky="w", pady=5
        )  # Tạo nhãn cho mỗi trường

        # Tạo ô nhập liệu tương ứng cho từng nhãn
        entry = tk.Entry(
            form_frame, font=("Arial", 12)
        )  # Tạo ô nhập liệu cho mỗi trường
        entry.grid(row=i * 2, column=1, pady=5)  # Đặt ô nhập liệu bên cạnh nhãn
        entries[label] = entry  # Lưu trữ ô nhập liệu trong dictionary

        # Tạo nhãn lỗi, ban đầu để trống
        error_label = tk.Label(form_frame, text="", fg="red")  # Tạo nhãn lỗi
        error_label.grid(
            row=i * 2 + 1, column=1, pady=5, sticky="w"
        )  # Đặt nhãn lỗi ngay dưới ô nhập liệu
        error_labels[label] = error_label  # Lưu nhãn lỗi vào dictionary

    # Hàm xử lý khi nhấn nút "Dự báo"
    def submit_form():
        # Lấy dữ liệu người dùng nhập vào
        user_data = {label: entry.get() for label, entry in entries.items()}

        # Làm sạch dữ liệu
        cleaned_data, error_found = clean_input_data(user_data, error_labels)

        if error_found:  # Nếu có lỗi
            result_label.config(
                text="Dữ liệu không hợp lệ! Vui lòng kiểm tra các trường có lỗi."
            )
            return  # Dừng thực hiện nếu dữ liệu không hợp lệ

        # Lưu dữ liệu sạch vào file CSV
        df = pd.DataFrame([cleaned_data])  # Tạo DataFrame từ dữ liệu sạch
        input_path = get_relative_path(
            "ThoiTiet_input.csv"
        )  # Lấy đường dẫn tương đối của file
        df.to_csv(input_path, index=False)  # Ghi dữ liệu vào file CSV

        # Thực hiện dự đoán bằng thuật toán k-NN
        item = loadDataInput(input_path)  # Tải dữ liệu đầu vào
        knn = kNearestNeighbor(trainSet, item[0], 5)  # Lấy 5 hàng xóm gần nhất
        prediction = findMostOccur(knn)  # Tìm nhãn xuất hiện nhiều nhất
        translated_prediction = label_translation.get(
            prediction, prediction
        )  # Dịch nhãn sang tiếng Việt nếu cần

        # Hiển thị kết quả dự đoán trong giao diện
        result_label.config(
            text=f"Dự đoán: {translated_prediction}", font=("Arial", 14, "bold")
        )

        # Tải và hiển thị biểu tượng tương ứng với dự đoán
        icon = load_icon(prediction)  # Tải biểu tượng
        icon_label.config(image=icon)  # Gắn hình vào nhãn
        icon_label.image = icon  # Giữ tham chiếu đến hình ảnh để tránh bị xóa

        # Lưu kết quả dự đoán vào file CSV
        with open(
            get_relative_path("ThoiTiet_testInput.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.writer(f)  # Tạo writer để ghi file
            writer.writerow(["Input Label", "Predicted"])  # Ghi tiêu đề
            writer.writerow(
                [label_translation.get(item[0][-1], item[0][-1]), translated_prediction]
            )  # Ghi dữ liệu

        # Tính toán và hiển thị độ chính xác
        accuracy = calculate_accuracy(testSet, trainSet, k=5)  # Tính tỷ lệ chính xác
        accuracy_label.config(
            text=f"Tỷ lệ chính xác: {accuracy:.2f}%"
        )  # Hiển thị tỷ lệ chính xác

    # Nút "Dự Đoán" 
    submit_button = tk.Button(
        root,
        text="Dự Đoán",
        command=submit_form,
        bg="green",
        fg="white",
        font=("Arial", 12),
    )
    submit_button.pack(pady=10)  # Thêm khoảng cách bên dưới nút

    # Nhãn hiển thị kết quả dự đoán
    result_label = tk.Label(root, text="Dự đoán: Chưa có dự đoán.")
    result_label.pack(pady=10)

    # Nhãn hiển thị biểu tượng thời tiết dự đoán
    icon_label = tk.Label(root)
    icon_label.pack(pady=10)

    # Nhãn hiển thị độ chính xác
    accuracy_label = tk.Label(root, text="Tỷ lệ chính xác: Chưa tính toán.")
    accuracy_label.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    trainSet, testSet = loadData()
    create_gui()
