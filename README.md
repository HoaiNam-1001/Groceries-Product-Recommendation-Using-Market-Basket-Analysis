# 🛒 Groceries Product Recommendation Using Market Basket Analysis — FP-Growth

## Overview
Market Basket Analysis trên tập giao dịch bán lẻ thực tế gồm 176,474 giao dịch từ 6 loại cửa hàng. Sử dụng thuật toán **FP-Growth** để khai thác tập phổ biến và sinh luật kết hợp, từ đó xây dựng hàm gợi ý sản phẩm dựa trên **Confidence** và **Lift > 1**.

## Tech Stack
Python | mlxtend (FP-Growth) | Pandas | NumPy | Plotly | Matplotlib | Seaborn

## Dataset
File `Retail_Transactions_Dataset2.csv` (~27 MB) không có trong repo.  
📥 **Tải tại:** [YOUR_DRIVE_LINK_HERE]

Sau khi tải, đặt file vào thư mục gốc của project.

| Thông tin | Giá trị |
|---|---|
| Kích thước | 176,474 giao dịch × 13 cột |
| Loại cửa hàng | Supermarket, Pharmacy, Department Store, Warehouse Club, Specialty Store, Convenience Store |
| Thành phố | 10 thành phố |
| Mùa | Winter, Spring, Summer, Fall |
| Phương thức thanh toán | Cash, Credit Card, Debit Card, Mobile Payment |

## Results

| Tham số | Giá trị |
|---|---|
| min_support | 0.002 |
| min_confidence | 0.01 |
| Lọc luật | lift > 1 |
| Sample training | 20,000 giao dịch (random_state=42) |

Gợi ý sản phẩm dựa trên confidence cao nhất trong các luật có chứa sản phẩm đầu vào:
```python
product_recommendations('Toothpaste')
# → Top 10 sản phẩm thường mua kèm, sắp xếp theo confidence giảm dần
```

## How to Run
```
pip install -r requirements.txt
jupyter notebook final_khai_pha__1_.ipynb
```

**`requirements.txt`:**
```
pandas
numpy
matplotlib
seaborn
plotly
mlxtend
pyforest
jupyter
```
