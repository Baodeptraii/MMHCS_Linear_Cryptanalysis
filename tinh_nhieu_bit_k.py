import des
import random
from collections import defaultdict
from functools import reduce
import operator

def xor_bits(bits):
    return reduce(operator.xor, bits)

def linear_attack_multiple_sboxes(pairs, sbox_info_list):
    """
    Thực hiện tấn công tuyến tính trên nhiều S-Box
    pairs: danh sách các cặp (plaintext, ciphertext)
    sbox_info_list: danh sách thông tin S-Box cần tấn công
    """
    results = {}
    
    for sbox_info in sbox_info_list:
        sbox_num = sbox_info['sbox_num']
        input_mask = sbox_info['input_mask']
        output_mask = sbox_info['output_mask']
        bias = sbox_info['bias']
        key_bits = sbox_info['key_bits']
        
        counts = defaultdict(int)
        
        for pt, ct in pairs:
            pt_bits = des.bytes_to_bits(pt)
            ct_bits = des.bytes_to_bits(ct)
            
            # Áp dụng hoán vị ban đầu và hoán vị cuối ngược
            pt_ip = [pt_bits[x] for x in des.IP]
            ct_fp_inv = [ct_bits[x] for x in des.inverse_FP]
            
            # Lấy nửa trái và phải của plaintext và ciphertext
            pl = pt_ip[32:]  # nửa phải plaintext (sau IP)
            cl = ct_fp_inv[32:]  # nửa phải ciphertext (sau FP^-1)
            
            # Tính input bits từ input mask
            input_bits = []
            for i, bit in enumerate(bin(input_mask)[2:].zfill(6)):
                if bit == '1':
                    input_bits.append(pl[des.E[(sbox_num-1)*6 + i]])
            
            # Tính output bits từ output mask
            output_bits = []
            for i, bit in enumerate(bin(output_mask)[2:].zfill(4)):
                if bit == '1':
                    output_bits.append(cl[des.P.index((sbox_num-1)*4 + i)])
            
            # Tính giá trị của biểu thức tuyến tính
            linear_expr = xor_bits(input_bits + output_bits)
            
            # Đếm số lần biểu thức tuyến tính bằng 0
            if linear_expr == 0:
                counts['zero'] += 1
            else:
                counts['one'] += 1
        
        # Tính bias thực tế
        actual_bias = (counts['zero'] - counts['one']) / len(pairs)
        
        # Dự đoán giá trị của XOR các bit khóa
        if (actual_bias > 0 and bias > 0) or (actual_bias < 0 and bias < 0):
            key_xor_guess = 0
        else:
            key_xor_guess = 1
        
        results[sbox_num] = {
            'zero_count': counts['zero'],
            'one_count': counts['one'],
            'actual_bias': actual_bias,
            'key_xor_guess': key_xor_guess,
            'key_bits': key_bits
        }
    
    return results

def generate_pairs(des_instance, num_pairs):
    """Tạo các cặp plaintext-ciphertext ngẫu nhiên"""
    pairs = []
    for _ in range(num_pairs):
        pt = bytes([random.getrandbits(8) for _ in range(8)])
        ct = des_instance.encrypt(pt)
        pairs.append((pt, ct))
    return pairs

if __name__ == "__main__":
    # Khởi tạo DES với 5 vòng (để phù hợp với tấn công tuyến tính)
    key = bytes([random.getrandbits(8) for _ in range(8)])
    des_instance = des.DES(key=key, round_num=5)
    
    print(f"Khóa thực (ẩn): {key.hex()}")
    
    # Danh sách các S-Box và thông tin linear approximation cần tấn công
    sbox_info_list = [
        {
            'sbox_num': 3,
            'input_mask': 0b101111,
            'output_mask': 0b1000,
            'bias': 14,
            'key_bits': [12,14,15,16,17]
        },
        {
            'sbox_num': 4,
            'input_mask': 0b111001,
            'output_mask': 0b0011,
            'bias': 12,
            'key_bits': [18,19,20,23]
        },
        {
            'sbox_num': 5,
            'input_mask': 0b001100,
            'output_mask': 0b0111,
            'bias': 10,
            'key_bits': [26,27]
        },
        {
            'sbox_num': 6,
            'input_mask': 0b111101,
            'output_mask': 0b1000,
            'bias': 12,
            'key_bits': [30,31,32,33,35]
        },
        {
            'sbox_num': 8,
            'input_mask': 0b111000,
            'output_mask': 0b0001,
            'bias': 28,
            'key_bits': [42,43,44]
        }
    ]
    
    # Tạo 65536 cặp plaintext-ciphertext (theo đề xuất của Matsui)
    pairs = generate_pairs(des_instance, 65536)
    
    # Thực hiện tấn công tuyến tính trên nhiều S-Box
    results = linear_attack_multiple_sboxes(pairs, sbox_info_list)
    
    # Hiển thị kết quả
    for sbox_num, result in results.items():
        print(f"\nKết quả S-Box {sbox_num}:")
        print(f"Biểu thức = 0: {result['zero_count']} lần")
        print(f"Biểu thức = 1: {result['one_count']} lần")
        print(f"Bias thực tế: {result['actual_bias']:.4f}")
        print(f"Dự đoán XOR(K{result['key_bits']}) = {result['key_xor_guess']}")
    
    # Phân tích kết hợp các kết quả
    print("\nPhân tích tổng hợp:")
    # Có thể thêm logic kết hợp các kết quả từ nhiều S-Box để thu được nhiều bit khóa hơn
    # Ví dụ: nếu có đủ thông tin, có thể giải hệ phương trình để tìm các bit khóa riêng lẻ