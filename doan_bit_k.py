import random
import des  # sử dụng des.py đã cung cấp

'''
S-Box 8 (S8) được sử dụng trong DES có độ lệch tuyến tính là 28.
SBOX #8:
input mask: 111000, output mask: 0001, bias: 28
X[27,28,29] ^ F(X, K)[30] = K[42,43,44]
input mask: 111000, output mask: 0101, bias: 28
X[27,28,29] ^ F(X, K)[14,30] = K[42,43,44]

'''

def xor_bits(bits):
    return sum(bits) % 2

def linear_attack_xor_keybit(pairs, bias=28):
    """
    pairs: danh sách (plaintext, ciphertext)
    bias: độ lệch tuyến tính (default là 28 theo S-box 8)
    """
    N = len(pairs)
    count = 0

    for pt, ct in pairs:
        pt_bits = des.bytes_to_bits(pt)
        ct_bits = des.bytes_to_bits(ct)

        pt_ip = [pt_bits[x] for x in des.IP]          # IP
        ct_fp_inv = [ct_bits[x] for x in des.inverse_FP]  # inverse FP

        pl = pt_ip[32:]  # lower half
        cl = ct_fp_inv[32:]  # lower half sau FP^-1

        # X[27] ^ X[28] ^ X[29] ^ F(X)[30]
        result = xor_bits([pl[27], pl[28], pl[29], cl[30]])

        if result == 0:
            count += 1

    if count > N // 2:
        guess = 0 if bias > 0 else 1
    else:
        guess = 1 if bias > 0 else 0

    print(f"Biểu thức tuyến tính = 0: {count} / {N}")
    print(f"Đoán XOR(K[42], K[43], K[44]) = {guess}")
    return guess

def generate_plaintext_ciphertext_pairs(des_instance, N=65536):
    pairs = []
    for _ in range(N):
        pt = bytes(random.getrandbits(8) for _ in range(8))
        ct = des_instance.encrypt(pt)
        pairs.append((pt, ct))
    return pairs

if __name__ == "__main__":
    # Khởi tạo DES với 5 vòng như trong bài lab (Matsui's attack)
    # Khóa 8 byte ngẫu nhiên
    key = bytes([random.getrandbits(8) for _ in range(8)])
    des_instance = des.DES(key=key, round_num=5)

    print(f"Khóa đang dùng (ẩn): {key.hex()}")

    pairs = generate_plaintext_ciphertext_pairs(des_instance)
    linear_attack_xor_keybit(pairs)
'''
Output:
Khóa đang dùng (ẩn): 71f9cc0ce76f042e
Biểu thức tuyến tính = 0: 32698 / 65536 ~ 49.89%
Đoán XOR(K[42], K[43], K[44]) = 1
'''