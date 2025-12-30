# Dữ liệu Dự án (Project Datasets)

Thư mục này chứa các tệp dữ liệu phục vụ cho dự án **Heart Disease Risk Screening**.

## ƯU Ý QUAN TRỌNG (IMPORTANT NOTICE)

Do chính sách giới hạn kích thước tệp của GitHub (>100MB) và để tối ưu hóa hiệu suất repository, **dữ liệu gốc đầy đủ (Full Dataset) KHÔNG được lưu trữ trực tiếp tại đây**.

Chúng tôi chỉ cung cấp tệp mẫu (`sample_data.csv`) chứa 50 dòng đầu tiên để minh họa cấu trúc dữ liệu và định dạng cột phục vụ việc kiểm thử mã nguồn (Sanity Check).

---

## Danh sách tệp tin (File Inventory)

| Tên File | Loại | Mô tả | Trạng thái |
| :--- | :--- | :--- | :--- |
| **`sample_data.csv`** | `CSV` | Dữ liệu mẫu (50 dòng). Dùng để kiểm tra code chạy thử. | ✅ Có sẵn |
| **`heart_2022_no_nans.csv`** | `CSV` | Dữ liệu gốc đã xử lý NaN (~300MB). Dùng để Huấn luyện (Train). | ❌ **Cần tải về** |
| `README.md` | `MD` | Tài liệu hướng dẫn thiết lập dữ liệu. | ✅ Có sẵn |

---

## Hướng dẫn thiết lập (Setup Instructions)

Để thực thi quy trình huấn luyện (`app/train.py`) hoặc chạy Notebook phân tích (`demo/HeartDisease_BRFSS.ipynb`), vui lòng thực hiện đúng 3 bước sau:

### Bước 1: Tải xuống (Download)
Truy cập đường dẫn bên dưới để tải bộ dữ liệu gốc:
> **🔗 Link tải:** [https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease]

### Bước 2: Đổi tên (Rename)
Sau khi tải về, hãy đảm bảo đổi tên file chính xác để khớp với mã nguồn:
* Tên file tải về (dự kiến): `heart_2022.csv` (hoặc tên mặc định từ nguồn)
* **Đổi tên thành:** `heart_2022_no_nans.csv`

### Bước 3: Di chuyển (Move)
Di chuyển file `heart_2022_no_nans.csv` vào chính thư mục `data/` này.

---

## Kiểm tra cấu trúc (Verification)

Sau khi hoàn tất, cấu trúc thư mục trên máy cục bộ (Local Machine) phải như sau:

```text
HeartDisease_Project/
├── data/
│   ├── README.md
│   ├── sample_data.csv
│   └── heart_2022_no_nans.csv  <-- (File này BẮT BUỘC phải có để Train)