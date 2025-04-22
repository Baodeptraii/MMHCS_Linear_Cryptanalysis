import numpy as np
from collections import Counter

# Cài đặt S-box số 5 (mô phỏng để minh họa)
def sbox5(x):
    # Đây là hàm S-box giả lập
    # Trong một triển khai thực tế, đây sẽ là ánh xạ S-box thực tế
    # Ở ví dụ này, ta sử dụng một phép biến đổi đơn giản
    return (x * 3 + 5) % 16

# Hàm áp dụng một mặt nạ vào giá trị
def apply_mask(value, mask):
    # Tính toán độ chẵn lẻ (parity) của các bit được chọn bởi mặt nạ
    result = 0
    for i in range(32):
        if (mask >> i) & 1:
            result ^= (value >> i) & 1
    return result

# Tấn công phá mã tuyến tính
def linear_attack(plaintext_blocks, ciphertext_blocks, input_mask, output_mask, num_key_bits=8):
    """
    Thực hiện tấn công phá mã tuyến tính để khôi phục các bit khóa
    
    Các tham số:
        plaintext_blocks: Danh sách các khối plaintext
        ciphertext_blocks: Danh sách các khối ciphertext tương ứng
        input_mask: Mặt nạ đầu vào cho biểu thức tuyến tính
        output_mask: Mặt nạ đầu ra cho biểu thức tuyến tính
        num_key_bits: Số bit khóa cần khôi phục
        
    Trả về:
        Giá trị khóa có xác suất cao nhất
    """
    # Khởi tạo bộ đếm cho mỗi giá trị khóa có thể có
    key_counts = Counter()
    total_samples = len(plaintext_blocks)
    
    # Thử mỗi giá trị khóa có thể có
    for key_guess in range(2**num_key_bits):
        correct_count = 0
        
        for p, c in zip(plaintext_blocks, ciphertext_blocks):
            # Áp dụng dự đoán khóa để có đầu vào cho S-box
            s_input = p ^ key_guess
            
            # Áp dụng S-box
            s_output = sbox5(s_input)
            
            # Tính độ chẵn lẻ cho biểu thức tuyến tính
            input_parity = apply_mask(s_input, input_mask)
            output_parity = apply_mask(s_output, output_mask)
            
            # Nếu biểu thức tuyến tính đúng, tăng bộ đếm
            if input_parity == output_parity:
                correct_count += 1
        
        # Tính độ lệch (bias) cho dự đoán khóa này
        bias = abs((correct_count / total_samples) - 0.5)
        key_counts[key_guess] = bias
    
    # Trả về khóa có độ lệch cao nhất
    return key_counts.most_common(1)[0][0], key_counts

# Hàm demo thực hiện toàn bộ quá trình tấn công
def demonstrate_linear_attack():
    # Các tham số từ bài toán
    input_mask = 16  # 10000 trong hệ nhị phân
    output_mask = 15  # 01111 trong hệ nhị phân
    known_bias = 0.31250
    # Sinh dữ liệu giả lập
    np.random.seed(42)  # Để tạo ra kết quả có thể tái tạo
    true_key = 0b11010010  # Khóa 8-bit cho minh họa
    num_samples = 65536
    print(f"Khóa thật: {true_key:08b} (thập phân: {true_key})")
    print(f"Độ lệch dự kiến cho khóa đúng: {known_bias}")
    print(f"Số mẫu: {num_samples}")
    # Sinh các plaintext và ciphertext tương ứng
    plaintexts = [np.random.randint(0, 16) for _ in range(num_samples)]
    ciphertexts = []
    
    for p in plaintexts:
        # Áp dụng khóa
        s_input = p ^ (true_key & 0xF)  # Dùng 4 bit thấp nhất của khóa
        # Áp dụng S-box
        s_output = sbox5(s_input)
        # Trong một kịch bản thực tế, có thể có thêm các phép toán khác ở đây
        ciphertexts.append(s_output)
    
    recovered_key, key_counts = linear_attack(plaintexts, ciphertexts, input_mask, output_mask, num_key_bits=4)
    
    print(f"Khóa thu hồi được: {recovered_key:04b} (thập phân: {recovered_key})")
    print(f"Khóa thật (4 bit thấp nhất): {true_key & 0xF:04b} (thập phân: {true_key & 0xF})")
    
    if recovered_key == (true_key & 0xF):
        print("Thành công! Các bit khóa đã được khôi phục chính xác.")
    else:
        print("Khóa thu hồi không khớp với khóa thật.")
        print("Điều này có thể do số mẫu không đủ hoặc do sự đơn giản trong ví dụ của chúng ta.")
                
    sorted_key_counts = sorted(key_counts.items(), key=lambda x: -x[1])
    print("\nĐộ lệch (bias) của các khóa dự đoán (giảm dần):")
    for key, bias in sorted_key_counts:
        tag = " <= Khóa đúng" if key == (true_key & 0xF) else ""
        print(f"Khóa {key:04b} có độ lệch {bias:.4f}{tag}")

if __name__ == "__main__":
    demonstrate_linear_attack()
