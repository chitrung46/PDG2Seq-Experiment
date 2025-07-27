# PDG2Seq with Hypergraph Integration

Tích hợp hypergraph vào mô hình PDG2Seq để xử lý dữ liệu giao thông với các loại hyperedge khác nhau.

## Cấu trúc Hypergraph

### Các loại Hyperedge

1. **Pick/Drop Similarity** (`pick_drop`): Tạo hyperedge giữa các node có giá trị pick và drop tương tự tại cùng thời điểm
2. **Geographical** (`geo`): Tạo hyperedge giữa các node gần nhau về mặt địa lý (dựa trên ma trận khoảng cách)
3. **Temporal Change** (`temporal`): Tạo hyperedge giữa các node có biến động pick/drop tương tự theo thời gian
4. **Correlation** (`correlation`): Tạo hyperedge giữa các node có tương quan pick/drop qua thời gian
5. **Temporal Pattern** (`pattern`): Tạo hyperedge giữa các node có mẫu biến động tương tự

### Cấu trúc file

```
PDG2Seq/
├── model/
│   ├── PDG2Seq_DGCN.py           # Chứa HyperedgeBuilder và PDG2Seq_HyperGCN
│   ├── PDG2SeqHyperCell.py       # Cell cho hypergraph
│   └── PDG2Seq_Hypergraph.py     # Main model với hypergraph
├── config_file/
│   ├── NYC-Bike_PDG2Seq_HyperGCN.conf
│   └── NYC-Taxi_PDG2Seq_HyperGCN.conf
├── run_hypergraph.py             # Script chạy training
└── demo_hypergraph.py            # Demo script
```

## Sử dụng

### 1. Demo nhanh
```bash
cd PDG2Seq
python demo_hypergraph.py
```

### 2. Training với hypergraph
```bash
cd PDG2Seq
python run_hypergraph.py --dataset NYC-Bike --model PDG2Seq_HyperGCN --mode train
```

### 3. Test model đã train
```bash
python run_hypergraph.py --dataset NYC-Bike --model PDG2Seq_HyperGCN --mode test
```

## Cấu hình

Trong file config (ví dụ: `NYC-Bike_PDG2Seq_HyperGCN.conf`):

```ini
[hypergraph]
use_hypergraph = True
hyperedge_types = pick_drop,geo,temporal,correlation,pattern
pick_drop_threshold = 0.9
geo_threshold = 0.1
temporal_threshold = 0.8
correlation_threshold = 0.7
pattern_threshold = 0.8
pattern_window = 5
```

### Tham số hypergraph:
- `use_hypergraph`: Bật/tắt hypergraph
- `hyperedge_types`: Các loại hyperedge cần sử dụng (ngăn cách bởi dấu phẩy)
- `*_threshold`: Ngưỡng để tạo hyperedge cho từng loại
- `pattern_window`: Kích thước cửa sổ cho temporal pattern

## Dữ liệu yêu cầu

### Dữ liệu pick/drop
- File `.h5` với keys:
  - `bike_pick`, `bike_drop` cho dữ liệu bike
  - `taxi_pick`, `taxi_drop` cho dữ liệu taxi

### Ma trận khoảng cách (tùy chọn)
- File `dis_bb.csv` chứa ma trận khoảng cách N×N được chuẩn hóa
- Chỉ cần cho hyperedge `geo`

## Kiến trúc Model

### HyperedgeBuilder
Class chứa các static method để xây dựng hyperedge:
- `load_pick_drop_data()`: Load dữ liệu từ file .h5
- `load_distance_matrix()`: Load ma trận khoảng cách
- `build_*_edges()`: Các hàm xây dựng từng loại hyperedge

### PDG2Seq_HyperGCN
Thay thế GCN thường bằng HypergraphConv từ PyTorch Geometric:
- Sử dụng `torch_geometric.nn.HypergraphConv`
- Tự động xây dựng và cache hyperedge
- Hỗ trợ batch processing

### PDG2Seq_Hypergraph
Main model kế thừa cấu trúc PDG2Seq gốc:
- Encoder-Decoder architecture
- Tích hợp hypergraph convolution
- Tương thích với config và dataloader hiện có

## Ưu điểm

1. **Tận dụng code có sẵn**: Dataloader, config, trainer đều được giữ nguyên
2. **Modular design**: Có thể bật/tắt từng loại hyperedge
3. **Flexible thresholds**: Điều chỉnh ngưỡng cho từng loại quan hệ
4. **Auto caching**: Hyperedge được cache để tránh tính toán lại
5. **Fallback mechanism**: Tự động tạo edges đơn giản nếu lỗi

## Dependencies

Cần cài thêm:
```bash
pip install torch-geometric
pip install scikit-learn
```

## Ghi chú

- Model sẽ tự động detect loại dataset (Bike/Taxi) từ tên dataset
- Nếu không có file distance matrix, hyperedge `geo` sẽ bị bỏ qua
- Có thể combine với dynamic graph hiện có bằng cách thêm adj matrix vào forward
