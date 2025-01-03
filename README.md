# Neural Network + Softmax Layer

**Đồ án môn Project 1 (GVHD: Vũ Thị Hương Giang)**

## Dự đoán nhiều nhãn theo mô hình mạng nơ ron

### Hướng dẫn sử dụng

#### Bước 1: Nhập dữ liệu
1. Người dùng sẽ gửi lên 3 file bao gồm:
   - **Train file**: Sử dụng để huấn luyện.
   - **Validation file**: Để đánh giá khả năng của mô hình.
   - **Test file**: Nhằm kiểm tra hiệu quả của mô hình.
2. Sau khi gửi file, chương trình sẽ kiểm tra định dạng và tính toàn vẹn của dữ liệu.
3. Sau đó, chọn **"Analyze"**. Chương trình sẽ thực hiện phân tích file huấn luyện để đưa ra thông tin về tất cả các cột có trong file.

#### Bước 2: Chọn tham số
1. Người dùng thực hiện:
   - Chọn các **features**.
   - Chọn **target**.
   - Cấu trúc các lớp mạng ẩn.
   - Chọn các tham số huấn luyện mô hình.
2. Chọn **"Train"** để bắt đầu huấn luyện.
3. Sau khi hoàn tất, chương trình sẽ lưu lại các tham số mô hình và trả về phần trăm dự đoán đúng trên các tập dữ liệu.

#### Bước 3: Dự đoán
1. Chuyển sang trang **test**.
2. Thực hiện tải lên file muốn dự đoán.
3. Chọn **"Predict"**. Chương trình sẽ trả về một file Excel chứa các dự đoán của mô hình vừa được huấn luyện.
