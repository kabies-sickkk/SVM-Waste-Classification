import os
import random
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Hàm load và trích xuất đặc trưng từ ảnh ---
def load_images_from_folder(folder, label):
    images = []
    labels = []
    original_images = []  # Lưu ảnh gốc để hiển thị sau này
    for filename in tqdm(os.listdir(folder), desc=f"Loading {folder}"):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh xám ngay từ đầu
        if img is not None:
            img = cv2.resize(img, (64, 64))
            original_images.append(img)  # Lưu ảnh gốc
            img = cv2.GaussianBlur(img, (3, 3), 0)  # Giảm nhiễu bằng Gaussian Blur
            features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            images.append(features)
            labels.append(label)
    return images, labels, original_images

# --- Load dữ liệu train ---
plastic_images, plastic_labels, plastic_originals = load_images_from_folder(
    r"D:\tai_lieu_dai_hoc\THI_GIAC_MAY_TINH\Waste_Classification\dataset\train1\plastic", 0)
metal_images, metal_labels, metal_originals = load_images_from_folder(
    r"D:\tai_lieu_dai_hoc\THI_GIAC_MAY_TINH\Waste_Classification\dataset\train1\metal", 1)

# --- Gộp dữ liệu ---
X = np.array(plastic_images + metal_images)
y = np.array(plastic_labels + metal_labels)
original_images = plastic_originals + metal_originals  # Gộp danh sách ảnh gốc

# --- Hiển thị tổng số mẫu ảnh ---
print(f"\n📸 Tổng số mẫu ảnh: {len(X)}")

# --- Biểu đồ phân bố lớp ---
plt.figure(figsize=(8, 5))
sns.countplot(x=y)
plt.title('Phân bố các lớp trong tập dữ liệu')
plt.xlabel('Lớp')
plt.ylabel('Số lượng mẫu')
plt.xticks(ticks=[0, 1], labels=["Plastic (0)", "Metal (1)"])
plt.savefig('class_distribution.png')
plt.show()

# --- Chia dữ liệu train/test ---
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(X, y, original_images, test_size=0.2, random_state=42)
print(f"🔹 Số mẫu train: {len(X_train)}")
print(f"🔹 Số mẫu test: {len(X_test)}")

# --- Chuẩn hóa dữ liệu ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- So sánh độ chính xác với các giá trị gamma khác nhau ---
gammas = [15e-5, 25e-5, 75e-5, 125e-5, 135e-5]
accuracies = []

for gamma in gammas:
    print(f"\n Đang huấn luyện với gamma = {gamma}...")
    svm_model = SVC(kernel='rbf', C=100, gamma=gamma)
    svm_model.fit(X_train, y_train)
    
    # Đánh giá trên tập test
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# --- Vẽ biểu đồ cột so sánh độ chính xác ---
plt.figure(figsize=(8, 6))
plt.bar([str(g) for g in gammas], accuracies, color='skyblue')
plt.title('So sánh độ chính xác trên tập kiểm tra với các giá trị gamma khác nhau')
plt.xlabel('Gamma')
plt.ylabel('Độ chính xác (%)')
plt.ylim(0.90, 1)  # Đặt giới hạn trục tung từ 0 đến 1 (tương ứng với 0% đến 100%)
plt.tight_layout()
plt.savefig('gamma_comparison.png')
plt.show()

# In kết quả độ chính xác cho từng giá trị gamma
for gamma, accuracy in zip(gammas, accuracies):
    print(f"Gamma: {gamma}, Độ chính xác: {accuracy * 100:.2f}%")

# --- Huấn luyện mô hình với gamma tốt nhất ---
best_gamma = gammas[np.argmax(accuracies)]
print(f"\n Gamma tốt nhất là: {best_gamma} với độ chính xác {max(accuracies) * 100:.2f}%")

# --- Huấn luyện lại mô hình với gamma tốt nhất và đánh giá ---
svm_model = SVC(kernel='rbf', C=100, gamma=best_gamma)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n Độ chính xác trên tập test với gamma tốt nhất ({best_gamma}): {accuracy * 100:.2f}%")
print("\n Báo cáo phân loại:")
print(classification_report(y_test, y_pred))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Plastic (0)", "Metal (1)"], yticklabels=["Plastic (0)", "Metal (1)"])
plt.title('Confusion Matrix')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.savefig('confusion_matrix.png')
plt.show()

# --- Hiển thị 12 ảnh test ---
def display_predictions(img_test, y_test, y_pred, num_images=12):
    labels = {0: "Plastic", 1: "Metal"}
    plt.figure(figsize=(12, 8))
    for i in range(min(num_images, len(img_test))):
        plt.subplot(3, 4, i + 1)
        plt.imshow(img_test[i], cmap='gray')  # Hiển thị ảnh gốc
        color = 'green' if y_test[i] == y_pred[i] else 'red'
        plt.title(f'Thực: {labels[y_test[i]]}\nDự đoán: {labels[y_pred[i]]}', color=color)
        plt.axis('off')
    plt.suptitle('12 Ảnh Test (Xanh: Đúng, Đỏ: Sai)')
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()

display_predictions(img_test, y_test, y_pred)

# --- Hiển thị tất cả ảnh bị nhận diện sai ---
def display_misclassified_images(img_test, y_test, y_pred):
    labels = {0: "Plastic", 1: "Metal"}
    misclassified_indices = [i for i in range(len(y_test)) if y_test[i] != y_pred[i]]
    if not misclassified_indices:
        print("Không có ảnh nào bị nhận diện sai.")
        return
    
    num_images = len(misclassified_indices)
    rows = (num_images // 4) + (num_images % 4 > 0)  # Số hàng cần để hiển thị hết ảnh
    
    plt.figure(figsize=(12, rows * 3))  # Điều chỉnh kích thước theo số hàng
    for i, idx in enumerate(misclassified_indices):
        plt.subplot(rows, 4, i + 1)
        plt.imshow(img_test[idx], cmap='gray')  # Hiển thị ảnh bị nhận diện sai
        plt.title(f'Thực: {labels[y_test[idx]]}\nDự đoán: {labels[y_pred[idx]]}', color='red')
        plt.axis('off')
    
    plt.suptitle(f'Tổng số ảnh bị nhận diện sai: {num_images}')
    plt.tight_layout()
    plt.savefig('all_misclassified_images.png')
    plt.show()

display_misclassified_images(img_test, y_test, y_pred)

# --- Kết thúc chương trình ---
