import des
import operator
from functools import reduce
import sympy
import random
import os
import math

# Hàm sử lý S-box và bias
 
'''Mục đích: mô phỏng hoạt động của một S-box.
Nhận vào một số 6 bit i, chọn dòng/ cột tương ứng để tra bảng S-box và trả về giá trị 4 bit đầu ra.'''
def _sub(i, S_BOX):
    block = bin(i)[2:].zfill(6)
    ret = []
    row = int(str(block[0])+str(block[5]), 2) #Get the row with the first and last bit
    column = int(''.join([str(x) for x in block[1:][:-1]]),2) #Column is the 2,3,4,5th bits
    val = S_BOX[row][column] #Take the value in the SBOX appropriated for the round (i)
    for j in range(3,-1,-1):
        ret.append((val >> j) & 1)            
    return int(''.join(map(str, ret)), 2) 
# Tính parity (bit chẵn lẻ XOR) của một số.
def xor_bits(x):
    return reduce(operator.xor, map(int, bin(x)[2:]))


#  Tính Bảng LAT (Linear Approximation Table)
'''
generate_LAT(S_BOX)
Tạo bảng LAT cho một S-box 6→4, với:
alpha: input mask (1 → 63).
beta: output mask (1 → 15).
Với mỗi cặp alpha, beta, đếm số lần xor_bits(x & alpha) == xor_bits(S(x) & beta), rồi trừ đi 32 (vì có 64 đầu vào, 
ta mong chờ mỗi biểu thức đúng 32 lần nếu hoàn toàn ngẫu nhiên → lệch khỏi 32 mới là bias).
Kết quả là ma trận 63x15 lưu các bias.'''
def generate_LAT(S_BOX):
    Lat = [[-32 for i in range(15)] for j in range(63)]
    for alpha in range(1, 64): # input mask from 1 to 63
        for beta in range(1, 16): # output mask from 1 to 16
            for x in range(64):
                if xor_bits(x & alpha) == xor_bits(_sub(x, S_BOX) & beta): # check if the input bits xorred equals the output bits xorred after substitution and increase the bias in the LAT if it does
                    Lat[alpha - 1][beta - 1] += 1
    return Lat


#Tìm các xấp xỉ tuyến tính mạnh nhất

'''find_linear_approx(lat_idx, THRESHOLD, ignore_indexes=None)
Duyệt toàn bộ LAT của một S-box (lat_idx) và in ra các bias có độ lệch lớn hơn ngưỡng THRESHOLD.
Nếu có ignore_indexes, thì loại bỏ những xấp xỉ liên quan đến các bit “xấu” (ví dụ: bit 5, 7 trong bài này).
'''
def find_linear_approx(lat_idx, THRESHOLD=14, ignore_indexes=None):
    lat = LATs[lat_idx]
    if ignore_indexes:
         for i in range(len(lat)):
            for j in range(len(lat[0])):
                if abs(lat[i][j]) >= THRESHOLD:
                    input_mask = i + 1
                    output_mask = j + 1
                    a = [des.E[lat_idx*6 + x] for x, y in enumerate(bin(input_mask)[2:].zfill(6)) if y == "1"] # input bits
                    b = [des.P.index(lat_idx*4 + x) for x, y in enumerate(bin(output_mask)[2:].zfill(4)) if y == "1"] # output bits
                    if any(thing in ignore_indexes for thing in a + b): continue
                    print(f"input mask: {bin(i + 1)[2:].zfill(6)}, output mask: {bin(j + 1)[2:].zfill(4)}, bias: {lat[i][j]}")
                    print_matsui_equation(lat_idx + 1, i + 1, j + 1)
    else:
        for i in range(len(lat)):
            for j in range(len(lat[0])):
                if abs(lat[i][j]) >= THRESHOLD:
                    print(f"input mask: {bin(i + 1)[2:].zfill(6)}, output mask: {bin(j + 1)[2:].zfill(4)}, bias: {lat[i][j]}")
                    print_matsui_equation(lat_idx + 1, i + 1, j + 1)

# In ra phương trình Matsui
# Phương trình này thể hiện mối quan hệ giữa các bit đầu vào, đầu ra và khóa trong một S-box.
def print_matsui_equation(sbox_num, input_mask, output_mask):
    a = [des.E[(sbox_num - 1)*6 + x] for x, y in enumerate(bin(input_mask)[2:].zfill(6)) if y == "1"] # input bits
    b = [des.P.index((sbox_num - 1)*4 + x) for x, y in enumerate(bin(output_mask)[2:].zfill(4)) if y == "1"] # output bits
    c = [(sbox_num - 1)*6 + x for x, y in enumerate(bin(input_mask)[2:].zfill(6)) if y == "1"] # key bits
    a, b, c = sorted(a), sorted(b), sorted(c)

    print(f"X[{','.join(map(str, a))}] ^ F(X, K)[{','.join(map(str, b))}] = K[{','.join(map(str, c))}]")

LATs = []
for i in range(8):
    LATs.append(generate_LAT(des.S_BOX[i]))

bad_bits = [5, 7]
bad_bits_full = [thing + 8*i for thing in bad_bits for i in range(8)]
bad_idxes = [des.IP.index(bad) for bad in bad_bits_full]
for i in range(8):
    print(f"SBOX #{i + 1}:")
    find_linear_approx(i, THRESHOLD=10, ignore_indexes=bad_idxes)
    print("--------------------------")

idxes = [2,5,7,9,10,11,12,22,26,16]
a = [des.IP[idx] for idx in idxes]
print(a)
a8 = [thing % 8 for thing in a]
print(a8)

''' Kết nối bias vào chuỗi xấp xỉ (chaining approximation)
find_linear_approx_chain(inp_bit_idx)
Cho một bit đầu ra từ F (đã được permute bởi P), hàm sẽ tìm các input mask có bias cao tương ứng → dùng để xây dựng chuỗi tuyến tính nhiều vòng.
'''
def find_linear_approx_chain(inp_bit_idx, THRESHOLD=10, ignore_indexes=None):
    inp_bit_idx = des.P[inp_bit_idx]

    sbox_num = inp_bit_idx // 4
    o = inp_bit_idx % 4
    output_mask = '0'*o + '1' + '0'*(3 - o) # assuming single index
    print(f"SBOX #{sbox_num + 1}:")
    print(f"Output mask: {output_mask}")

    output_mask = int(output_mask, 2)
    lat = LATs[sbox_num]

    for i in range(len(lat)):
        if abs(lat[i][output_mask - 1]) >= THRESHOLD:
            if ignore_indexes:
                input_mask = i + 1
                a = [des.E[sbox_num*6 + x] for x, y in enumerate(bin(input_mask)[2:].zfill(6)) if y == "1"] # input bits
                # output bits after sbox and xor is permuted using the P array
                b = [des.P.index(sbox_num*4 + x) for x, y in enumerate(bin(output_mask)[2:].zfill(4)) if y == "1"] # output bits
                if any(thing % 8 in ignore_indexes for thing in a + b): continue
                print(f"input mask: {bin(i + 1)[2:].zfill(6)}, output mask: {bin(output_mask)[2:].zfill(4)}, bias: {lat[i][output_mask - 1]}")
            else:
                print(f"input mask: {bin(i + 1)[2:].zfill(6)}, output mask: {bin(output_mask)[2:].zfill(4)}, bias: {lat[i][output_mask - 1]}")

'''Tính giá trị biểu thức tuyến tính trên plaintext/ciphertext'''

'''xor_multiple(arr, bits)
XOR nhiều bit từ mảng.'''
def xor_multiple(arr, bits):
    out = 0
    for bit in bits:
        out = arr[bit] ^ out
    return out

# Các biến toàn cục
K = sympy.IndexedBase('K') # key
PH = sympy.IndexedBase('PH') # upper half of input
PL = sympy.IndexedBase('PL') # lower half of input
CH = sympy.IndexedBase('CH') # upper half of output
CL = sympy.IndexedBase('CL') # lower half of output
X = sympy.IndexedBase('X') # intermediate input to found function F

def k(rnd_num, key_bits):
    return [K[rnd_num, key_bit] for key_bit in key_bits]

def round_func(rnd_num, input_bits, output_bits, key_bits):
    left = [*r(rnd_num, input_bits), *l(rnd_num, output_bits), *r(rnd_num+1, output_bits)]
    right = k(rnd_num, key_bits)
    return left, right

def l(rnd_num, bit_idxes):
    if rnd_num == 1: return [PH[bit_idx] for bit_idx in bit_idxes]
    return r(rnd_num - 1, bit_idxes)

def r(rnd_num, bit_idxes):
    if rnd_num == 1: return [PL[bit_idx] for bit_idx in bit_idxes]
    if rnd_num == 5: return [CL[bit_idx] for bit_idx in bit_idxes]
    if rnd_num == 6: return [CH[bit_idx] for bit_idx in bit_idxes]
    return x(rnd_num, bit_idxes)

def x(rnd_num, bit_idxes):
    return [X[rnd_num, bit_idx] for bit_idx in bit_idxes] 


def remove_duplicates(l, r):
    l_new = []
    for element in l:
        if l.count(element) % 2 != 0:
            l_new.append(element)
    
    r_new = []
    for element in r:
        if r.count(element) % 2 != 0:
            r_new.append(element)

    return l_new, r_new

alpha = [12,16]
beta = [27,28]
gamma = [42,43,44]

'''algorithm_1(pairs, bit_idxs, bias)
Với bit_idxs = [ph_bits, pl_bits, ch_bits, cl_bits], áp dụng IP và IP⁻¹ để đưa về không gian phân tích.
Tính giá trị biểu thức tuyến tính (theo kết quả bias), nếu nhiều hơn 50% mẫu thỏa mãn thì in ra khả năng bit khóa.
'''
def algorithm_1(pairs, bit_idxs, bias):
    N = len(pairs)
    ph_bits, pl_bits, ch_bits, cl_bits = bit_idxs

    cnt = 0
    for _pt, _ct in pairs:
        pt = des.bytes_to_bits(_pt)
        ct = des.bytes_to_bits(_ct)
        pt = [pt[x] for x in des.IP] # apply initial permutation
        ct = [ct[x] for x in des.inverse_FP] # apply inverse of final permutation

        ph, pl = pt[:32], pt[32:]
        ch, cl = ct[:32], ct[32:]

        T = xor_multiple(ph, ph_bits) ^ xor_multiple(pl, pl_bits) ^ xor_multiple(ch, ch_bits) ^ xor_multiple(cl, cl_bits)
        if T == 0:
            cnt += 1
    
    if cnt > N // 2:
        if bias > 0: return 0, cnt
        return 1, cnt
    elif bias > 0: return 1, cnt
    return 0, cnt

N = 65536
