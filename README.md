# SVM-waste-classification
Báo cáo tiểu luận Thị Giác Máy Tính và Ứng Dụng.
Đề tài: Ứng dụng máy vector hỗ trợ(SVM) trong phân loại rác thải: nhựa và kim loại
 
   Bài tiểu luận nghiên cứu ứng dụng Máy Vectơ Hỗ Trợ (SVM) trong việc phân
loại rác thải nhựa và kim loại, nhằm cải thiện hiệu quả trong quá trình phân loại
và tái chế. SVM, với khả năng tìm kiếm siêu phẳng tối ưu, đã được ứng dụng để
phân loại hình ảnh rác thải dựa trên các đặc trưng trích xuất từ phương pháp
Histogram of Oriented Gradients (HOG). Dữ liệu huấn luyện bao gồm 2000
hình ảnh chai nhựa và lon kim loại, được tiền xử lý và chuẩn hóa để cải thiện
hiệu suất mô hình. Sau khi thử nghiệm các tham số, mô hình đạt độ chính xác
cao, lên tới 97,38% trên tập kiểm tra. Tuy nhiên, một số hình ảnh vẫn bị phân
loại sai do nhiễu và ánh sáng không đồng đều. Bài viết cũng đề xuất các biện
pháp cải thiện như điều chỉnh kích thước ảnh, bổ sung dữ liệu huấn luyện và áp
dụng các kỹ thuật trích xuất đặc trưng mạnh mẽ hơn. Việc ứng dụng AI và học
máy trong phân loại rác thải có thể giúp tự động hóa quy trình xử lý và đóng
góp vào việc bảo vệ môi trường.
![image](https://github.com/user-attachments/assets/8f81de7a-d544-4b2f-93e9-bef978d08113)
- Hình ảnh hiện ngẫu nhiên 12 ảnh trong tập kiểm tra để thử nghiệm độ chính xác của mô hình, như ta thấy hầu hết điều nhận diện chính xác.Mô hình đạt độ chính xác 97,38% trên tập kiểm tra.

**hình ảnh phân loại sai:**

![image](https://github.com/user-attachments/assets/5f4821f2-8e23-4839-9087-99699a2984ed)

- đây là các hình ảnh bị nhận diện sai cụ thể là 23 bức ảnh, nguyên nhân có thể là do các bức ảnh bị nhiễu hoặc ánh sáng không đồng đều
ở đây ta có thể thấy trong hình có vài bức ảnh bị đổ bóng do máy ảnh, một phần là do kích thước của ảnh đã resize nhỏ nên vài chi tiết đã bị mất và bức ảnh không rõ  chi tiết nên dễ bị nhầm lẫn.

 **Ma trận nhầm lẫn giúp đánh giá số lượng mẫu bị phân loại sai và đúng:**
  ![image](https://github.com/user-attachments/assets/2f809894-8c5e-4315-acfb-8d82cf324189)

 **Kết luận**
 
Mô hình SVM đạt độ chính xác 97% trong phân loại rác thải nhựa và kim loại,
tuy nhiên vẫn có một số lỗi phân loại. Độ chính xác có thể cải thiện bằng cách
tối ưu siêu tham số (C, gamma, kernel), mở rộng tập dữ liệu và áp dụng các
phương pháp trích xuất đặc trưng tốt hơn như SIFT, ORB.
Việc ứng dụng AI vào phân loại rác giúp tự động hóa và tối ưu hóa quy trình xử
lý rác thải. Hệ thống này có thể tích hợp vào robot phân loại rác hoặc camera
giám sát thông minh để nhận diện và phân loại rác theo thời gian thực. Trong
tương lai, việc kết hợp Deep Learning (CNN, ViT) và phát triển ứng dụng hỗ
trợ nhận diện rác có thể giúp nâng cao hiệu quả thu gom và tái chế, góp phần
bảo vệ môi trường
