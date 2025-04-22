# CODE NÀY ĐỂ KIỂM TRA BIAS CỦA S-BOX TRONG THUẬT TOÁN DES
# Mô tả: Mã này tính toán bias cho tất cả các cặp (input_mask, output_mask) của từng S-box trong thuật toán DES.

# E là ma trận mở rộng, dùng để mở rộng nửa bên phải của khối dữ liệu 32 bit thành 48 bit.
E = [    31, 0,  1,  2,  3,  4,
    3,  4,  5,  6,  7,  8,
    7,  8,  9,  10, 11, 12,
    11, 12, 13, 14, 15, 16,
    15, 16, 17, 18, 19, 20,
    19, 20, 21, 22, 23, 24,
    23, 24, 25, 26, 27, 28,
    27, 28, 29, 30, 31, 0,]

# SBOX là ma trận thay thế, dùng để thay thế các bit đầu vào 6 bit thành 4 bit.
# Mỗi S-box có 4 hàng và 16 cột, với mỗi hàng đại diện cho một giá trị đầu vào (0-3) và mỗi cột đại diện cho một giá trị đầu ra (0-15).
S_BOX = [         
[[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
 [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
 [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
 [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
],

[[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
 [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
 [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
 [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
],

[[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
 [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
 [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
 [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
],

[[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
 [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
 [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
 [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
],  

[[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
 [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
 [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
 [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
], 

[[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
 [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
 [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
 [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
], 

[[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
 [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
 [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
 [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
],

[[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
 [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
 [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
 [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
]
]

# SHIFT là danh sách các giá trị dịch cho mỗi vòng trong quá trình tạo khóa.
SHIFT = [1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]

def parity(x):
    """Tính parity của một số nguyên (số bit 1 modulo 2)."""
    return bin(x).count('1') % 2

def sbox_output(sbox, input6bit):
    """Trả về output 4 bit sau khi đi qua S-box."""
    row = ((input6bit & 0b100000) >> 4) | (input6bit & 1)
    col = (input6bit >> 1) & 0b1111
    return sbox[row][col]

def linear_bias_for_sbox(sbox):
    """Tính bias cho tất cả cặp (input_mask, output_mask) của 1 S-box."""
    biases = {}
    for input_mask in range(1, 64):  # không tính mask 0
        for output_mask in range(1, 16):  # 4-bit output
            count = 0
            for input_val in range(64):
                output_val = sbox_output(sbox, input_val)
                in_parity = parity(input_mask & input_val)
                out_parity = parity(output_mask & output_val)
                if in_parity == out_parity:
                    count += 1
            prob = count / 64
            bias = abs(prob - 0.5)
            biases[(input_mask, output_mask)] = bias
    return biases

# Tính bias lớn nhất cho từng S-Box
for i, sbox in enumerate(S_BOX):
    biases = linear_bias_for_sbox(sbox)
    max_bias = max(biases.values())
    best_masks = [m for m, b in biases.items() if b == max_bias]
    print(f"SBOX {i+1}: max bias = {max_bias:.5f}, best (in_mask, out_mask) = {best_masks}")
