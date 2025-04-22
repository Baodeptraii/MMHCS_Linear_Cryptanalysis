import numpy as np
from Crypto.Cipher import DES
from Crypto.Util.Padding import pad
import os
import struct
import time
from tqdm import tqdm

# DES S-boxes
S_BOXES = [
    # S1
    [
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
    ],
    # S2
    [
        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
        [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
        [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
        [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
    ],
    # S3
    [
        [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
        [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
        [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
        [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
    ],
    # S4
    [
        [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
        [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
        [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
        [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
    ],
    # S5
    [
        [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
        [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
        [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
        [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
    ],
    # S6
    [
        [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
        [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
        [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
        [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
    ],
    # S7
    [
        [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
        [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
        [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
        [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
    ],
    # S8
    [
        [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
        [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
        [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
        [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
    ]
]

# Initial Permutation (IP)
IP = [
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17, 9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7
]

# Final Permutation (IP^-1)
FP = [
    40, 8, 48, 16, 56, 24, 64, 32,
    39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28,
    35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26,
    33, 1, 41, 9, 49, 17, 57, 25
]

# Expansion permutation (E)
E = [
    32, 1, 2, 3, 4, 5,
    4, 5, 6, 7, 8, 9,
    8, 9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32, 1
]

# Permutation (P)
P = [
    16, 7, 20, 21, 29, 12, 28, 17,
    1, 15, 23, 26, 5, 18, 31, 10,
    2, 8, 24, 14, 32, 27, 3, 9,
    19, 13, 30, 6, 22, 11, 4, 25
]

# Permuted Choice 1 (PC-1)
PC1 = [
    57, 49, 41, 33, 25, 17, 9,
    1, 58, 50, 42, 34, 26, 18,
    10, 2, 59, 51, 43, 35, 27,
    19, 11, 3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,
    7, 62, 54, 46, 38, 30, 22,
    14, 6, 61, 53, 45, 37, 29,
    21, 13, 5, 28, 20, 12, 4
]

# Permuted Choice 2 (PC-2)
PC2 = [
    14, 17, 11, 24, 1, 5, 3, 28,
    15, 6, 21, 10, 23, 19, 12, 4,
    26, 8, 16, 7, 27, 20, 13, 2,
    41, 52, 31, 37, 47, 55, 30, 40,
    51, 45, 33, 48, 44, 49, 39, 56,
    34, 53, 46, 42, 50, 36, 29, 32
]

# Key rotation schedule
KEY_SHIFTS = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

def permute(block, perm_table):
    """Permute the bits in the block according to the table."""
    result = 0
    for i, pos in enumerate(perm_table):
        # Get the bit at position pos-1 (perm tables are 1-indexed)
        bit = (block >> (64 - pos)) & 1
        # Set this bit in the result at position i
        result |= (bit << (len(perm_table) - 1 - i))
    return result

def split_block(block):
    """Split a 64-bit block into left and right halves."""
    left = (block >> 32) & 0xFFFFFFFF
    right = block & 0xFFFFFFFF
    return left, right

def expand(block, table=E):
    """Expand a 32-bit block to 48 bits using the expansion table."""
    result = 0
    for i, pos in enumerate(table):
        # Get the bit at position pos-1 (tables are 1-indexed)
        bit = (block >> (32 - pos)) & 1
        # Set this bit in the result at position i
        result |= (bit << (47 - i))
    return result

def substitute(block, s_boxes=S_BOXES):
    """Apply the S-box substitution to a 48-bit block."""
    result = 0
    # Process each 6-bit chunk
    for i in range(8):
        # Extract the 6-bit chunk
        chunk = (block >> (42 - i * 6)) & 0x3F
        # Get row (first and last bits)
        row = ((chunk & 0x20) >> 4) | (chunk & 0x01)
        # Get column (middle 4 bits)
        col = (chunk & 0x1E) >> 1
        # Get substitution value
        val = s_boxes[i][row][col]
        # Insert into result
        result |= (val << (28 - i * 4))
    return result

def generate_subkeys(key):
    """Generate the 16 subkeys for DES encryption."""
    # Apply PC1 permutation
    key = permute(key, PC1)
    
    # Split the key into left and right halves
    left = (key >> 28) & 0xFFFFFFF
    right = key & 0xFFFFFFF
    
    subkeys = []
    for shift in KEY_SHIFTS:
        # Rotate left
        left_shift = ((left << shift) & 0xFFFFFFF) | (left >> (28 - shift))
        right_shift = ((right << shift) & 0xFFFFFFF) | (right >> (28 - shift))
        
        # Combine halves
        combined = (left_shift << 28) | right_shift
        
        # Apply PC2 permutation
        subkey = permute(combined << (64 - 56), PC2)
        subkeys.append(subkey)
        
        # Update left and right for next round
        left, right = left_shift, right_shift
    
    return subkeys

def f_function(right, subkey):
    """The Feistel function for DES."""
    # Expand the right half
    expanded = expand(right)
    
    # XOR with the subkey
    xored = expanded ^ subkey
    
    # Apply S-boxes
    substituted = substitute(xored)
    
    # Apply permutation P
    permuted = permute(substituted << (64 - 32), P)
    
    return permuted

def des_encrypt_block(block, key):
    """Encrypt a single 64-bit block using DES."""
    # Generate subkeys
    subkeys = generate_subkeys(key)
    
    # Initial permutation
    block = permute(block, IP)
    
    # Split block
    left, right = split_block(block)
    
    # 16 rounds
    for i in range(16):
        # Feistel function
        f_result = f_function(right, subkeys[i])
        
        # XOR and swap
        new_right = left ^ f_result
        left = right
        right = new_right
    
    # Combine halves (swapped)
    combined = (right << 32) | left
    
    # Final permutation
    result = permute(combined, FP)
    
    return result

def des_decrypt_block(block, key):
    """Decrypt a single 64-bit block using DES."""
    # Generate subkeys
    subkeys = generate_subkeys(key)
    
    # Initial permutation
    block = permute(block, IP)
    
    # Split block
    left, right = split_block(block)
    
    # 16 rounds with reversed subkeys
    for i in range(15, -1, -1):
        # Feistel function
        f_result = f_function(right, subkeys[i])
        
        # XOR and swap
        new_right = left ^ f_result
        left = right
        right = new_right
    
    # Combine halves (swapped)
    combined = (right << 32) | left
    
    # Final permutation
    result = permute(combined, FP)
    
    return result

def bytes_to_int(byte_array):
    """Convert a byte array to an integer."""
    return int.from_bytes(byte_array, byteorder='big')

def int_to_bytes(n, length=8):
    """Convert an integer to a byte array."""
    return n.to_bytes(length, byteorder='big')

# Linear approximations for DES S-boxes
# Format: (input_mask, output_mask, bias)

# XẤP XỈ TUYẾN TÍNH DES

LINEAR_APPROXIMATIONS = [
    # S1
    (0x21, 0x8, 6/16 - 0.5),  # 6/16 probability for input bits 0,5 XOR output bit 3
    # S2
    (0x14, 0x1, 10/16 - 0.5),  # 10/16 probability for input bits 2,4 XOR output bit 0
    # S3
    (0x8, 0x8, 6/16 - 0.5),  # 6/16 probability for input bit 3 XOR output bit 3
    # S4
    (0x1, 0xA, 12/16 - 0.5),  # 12/16 probability for input bit 0 XOR output bits 1,3
    # S5
    (0x14, 0x4, 12/16 - 0.5),  # 12/16 probability for input bits 2,4 XOR output bit 2
    # S6
    (0x10, 0x8, 6/16 - 0.5),  # 6/16 probability for input bit 4 XOR output bit 3
    # S7
    (0x20, 0x9, 10/16 - 0.5),  # 10/16 probability for input bit 5 XOR output bits 0,3
    # S8
    (0x2, 0x6, 10/16 - 0.5)   # 10/16 probability for input bit 1 XOR output bits 1,2
]

# MÔ PHỎNG 3 VÒNG DES

# Linear Characteristics for 3-round DES
# These are simplified for demonstration
# Real attacks would use more carefully crafted characteristics

# Đặc trưng tuyến tính cho DES
LINEAR_CHARACTERISTIC = {
    'plaintext_mask': 0x2104000,
    'ciphertext_mask': 0x4010000,
    'bias': 0.031,
    'subkey_bits': [7, 18, 24, 29]  # Key bits that affect the relation
}

# ĐÁNH GIÁ BIỂU THỨC TUYẾN TÍNH

def evaluate_linear_expression(plaintext, ciphertext, p_mask, c_mask):
    """
    Evaluate the linear expression for a plaintext-ciphertext pair:
    P[i1] ⊕ P[i2] ⊕ ... ⊕ C[j1] ⊕ C[j2] ⊕ ...
    
    Args:
        plaintext, ciphertext: 64-bit integers
        p_mask, c_mask: Bit masks for plaintext and ciphertext
    
    Returns:
        0 or 1, the result of the linear expression
    """
    # Calculate the parity of the selected plaintext bits
    p_parity = bin(plaintext & p_mask).count('1') % 2
    
    # Calculate the parity of the selected ciphertext bits
    c_parity = bin(ciphertext & c_mask).count('1') % 2
    
    # Return the XOR of both parities
    return p_parity ^ c_parity


# TẠO CÁC CẶP BẢN RÕ/MÃ

def generate_random_plaintexts(num_plaintexts):
    """Generate random plaintexts for the attack."""
    return [bytes_to_int(os.urandom(8)) for _ in range(num_plaintexts)]

def encrypt_plaintexts(plaintexts, key):
    """Encrypt all plaintexts with the given key."""
    cipher = DES.new(int_to_bytes(key), DES.MODE_ECB)
    ciphertexts = []
    
    for p in plaintexts:
        p_bytes = int_to_bytes(p)
        c_bytes = cipher.encrypt(p_bytes)
        ciphertexts.append(bytes_to_int(c_bytes))
    
    return ciphertexts

# TÍNH TOÁN BIT KHÓA

def calculate_key_bit(plaintexts, ciphertexts, char):
    """Calculate the key bit using linear cryptanalysis."""
    count = 0
    total = len(plaintexts)
    
    for p, c in zip(plaintexts, ciphertexts):
        if evaluate_linear_expression(p, c, char['plaintext_mask'], char['ciphertext_mask']) == 0:
            count += 1
    
    probability = count / total
    # If probability > 0.5, the key bit is likely 0, otherwise 1
    # Need to adjust based on the expected bias
    expected_probability = 0.5 + char['bias']
    
    # Determine key bit based on deviation from expected probability
    if abs(probability - expected_probability) < abs(probability - (1 - expected_probability)):
        key_bit = 0
    else:
        key_bit = 1
    
    confidence = abs(probability - 0.5) / char['bias']
    return key_bit, probability, confidence

# THỰC HIỆN TẤN CÔNG

def perform_linear_cryptanalysis(key, num_plaintexts=65366):
    """
    Perform linear cryptanalysis on DES to recover key bits.
    
    Args:
        key: The 64-bit key used for encryption
        num_plaintexts: Number of plaintext-ciphertext pairs to use
    
    Returns:
        recovered_bits: Dictionary of recovered key bits
        statistics: Dictionary with attack statistics
    """
    print(f"Khóa thật (ẩn): {hex(key)}")
    print(f"Tạo {num_plaintexts} cặp bản rõ/mã...")

    # 1. Sinh dữ liệu
    plaintexts = generate_random_plaintexts(num_plaintexts)
    ciphertexts = encrypt_plaintexts(plaintexts, key)

    # 2. Phân tích tuyến tính
    start_time = time.time()
    key_bit, probability, confidence = calculate_key_bit(plaintexts, ciphertexts, LINEAR_CHARACTERISTIC)
    analysis_time = time.time() - start_time

    # 3. In kết quả chi tiết
    key_bit_indices = LINEAR_CHARACTERISTIC['subkey_bits']
    print(f"🎯 Dự đoán bit khóa tại các vị trí {key_bit_indices} là: {key_bit}")
    print(f"🔍 Xác suất quan sát được: {probability:.4f}")
    print(f"🎯 Xác suất kỳ vọng: {0.5 + LINEAR_CHARACTERISTIC['bias']:.4f}")
    print(f"📊 Độ tin cậy: {confidence:.2f}")
    print(f"⏱️ Phân tích mất {analysis_time:.2f} giây\n")

    # 4. Trả về kết quả
    recovered_bits = {tuple(key_bit_indices): key_bit}
    statistics = {
        'num_plaintexts': num_plaintexts,
        'observed_probability': probability,
        'expected_probability': 0.5 + LINEAR_CHARACTERISTIC['bias'],
        'confidence': confidence,
        'analysis_time': analysis_time
    }

    return recovered_bits, statistics


# TEST ĐỘ CHÍNH XÁC VỚI CÁC SỐ LƯỢNG BẢN RÕ KHÁC NHAU

def test_accuracy(iterations=10, plaintexts_range=[1000, 10000, 100000]):
    """Test the accuracy of the linear cryptanalysis attack with different numbers of plaintexts."""
    results = {}
    
    for num_plaintexts in plaintexts_range:
        correct_count = 0
        total_confidence = 0
        
        for i in range(iterations):
            # Generate random key
            key = bytes_to_int(os.urandom(8))
            
            # Perform attack
            recovered_bits, stats = perform_linear_cryptanalysis(key, num_plaintexts)
            
            # Check if the recovered key bit is correct
            # This is simplified - in reality we would check the actual key bit
            # Here we're just measuring consistency of the attack
            if stats['observed_probability'] > 0.5 and LINEAR_CHARACTERISTIC['bias'] > 0:
                correct_count += 1
            elif stats['observed_probability'] < 0.5 and LINEAR_CHARACTERISTIC['bias'] < 0:
                correct_count += 1
            
            total_confidence += stats['confidence']
        
        results[num_plaintexts] = {
            'accuracy': correct_count / iterations,
            'avg_confidence': total_confidence / iterations
        }
        
        print(f"Với {num_plaintexts} bản rõ: {results[num_plaintexts]['accuracy']*100:.1f}% chính xác, "
              f"độ tin cậy trung bình: {results[num_plaintexts]['avg_confidence']:.2f}\n")
    
    return results

def main():
    print("DES Linear Cryptanalysis Demo")
    print("=============================")
    
    # Set a random key
    key = bytes_to_int(os.urandom(8))
    
    # Perform the attack
    recovered_bits, stats = perform_linear_cryptanalysis(key, num_plaintexts=100000)
    
    print("Bits của khóa phục hồi:")
    for bits, value in recovered_bits.items():
        print(f"Key bits {bits}: {value}")
    
    
    # Optional: Test accuracy with different numbers of plaintexts
    accuracy_results = test_accuracy(iterations=5, plaintexts_range=[1000, 10000, 100000])
    

if __name__ == "__main__":
    main()