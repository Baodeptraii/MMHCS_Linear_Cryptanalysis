# IP, CP_1, CP_2, P, FP là các bảng hoán vị được sử dụng trong thuật toán DES.
IP = [3, 60, 61, 15, 41, 33, 13, 14, 53, 24, 36, 4, 8, 32, 22, 39, 43, 35, 12, 50, 30, 40, 20, 21, 0, 7, 34, 51, 57, 49, 52, 37, 6, 9, 48, 47, 19, 2, 16, 18, 26, 55, 62, 58, 31, 54, 11, 17, 10, 38, 56, 29, 1, 42, 28, 25, 5, 59, 27, 44, 46, 45, 23, 63]
CP_1 = [19, 37, 50, 47, 8, 4, 36, 30, 55, 6, 14, 35, 44, 0, 49, 41, 26, 12, 59, 53, 17, 54, 51, 57, 5, 22, 10, 2, 16, 52, 1, 62, 46, 60, 56, 33, 21, 29, 34, 48, 27, 38, 43, 18, 32, 9, 20, 61, 3, 13, 11, 15, 58, 45, 42, 28]
CP_2 = [19, 34, 14, 12, 3, 5, 42, 35, 17, 40, 27, 53, 29, 18, 7, 48, 0, 20, 22, 16, 28, 45, 39, 54, 1, 2, 38, 6, 31, 47, 44, 30, 52, 25, 43, 13, 49, 50, 24, 33, 15, 10, 26, 9, 21, 8, 23, 46]
P = [14, 13, 16, 26, 2, 19, 5, 22, 30, 27, 24, 0, 11, 9, 29, 3, 8, 15, 25, 12, 7, 23, 17, 21, 6, 28, 18, 4, 20, 10, 31, 1]
FP = [52, 48, 3, 46, 5, 49, 15, 37, 54, 41, 28, 4, 61, 40, 26, 24, 59, 34, 19, 27, 47, 21, 18, 43, 29, 9, 36, 35, 51, 62, 6, 45, 32, 12, 22, 56, 11, 16, 31, 50, 7, 10, 39, 2, 55, 14, 42, 13, 60, 0, 63, 25, 44, 8, 57, 20, 53, 1, 17, 38, 23, 58, 30, 33]
inverse_FP = [FP.index(i) for i in range(len(FP))]

##########################################################
##########################################################
########## E, S_BOX, SHIFT ARE NOT CHANGED!!!!! ##########
##########################################################
##########################################################

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
# bias = 28 
[
   [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1] 
]
]

# SHIFT là danh sách các giá trị dịch cho mỗi vòng trong quá trình tạo khóa.
SHIFT = [1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]

# Hàm này chuyển đổi một mảng byte thành một mảng bit.
def bytes_to_bits(bytes_arr):
    ret = []
    for b in bytes_arr:
        for i in range(7,-1,-1):
            ret.append((b >> i) & 1)
    return ret

# Hàm này chuyển đổi một mảng bit thành một mảng byte.
def bits_to_bytes(bits_arr): 
    assert len(bits_arr) % 8 == 0, "invalid bits_arr length"
    size = len(bits_arr) // 8
    ret = []
    for i in range(size):
        tmp = 0
        for j in range(8):
            tmp = 2 * tmp + bits_arr[8*i + j]
        ret.append(tmp)
    return bytes(ret)

# Hàm này thực hiện phép XOR giữa hai danh sách bit và trả về danh sách kết quả.
def xor(t1, t2):
    return [x^y for x,y in zip(t1,t2)]

class DES():
    def __init__(self, key, round_num):
        self.round_keys = []
        self.round_num = round_num
        self._keyschedule(key)

    # Hàm này tạo ra các khóa vòng từ khóa ban đầu.
    # Nó sử dụng bảng hoán vị CP_1 để tạo ra khóa 56 bit từ khóa 64 bit ban đầu.
    def _keyschedule(self, key):
        key = bytes_to_bits(key)
        key = [key[x] for x in CP_1]
        g, d = key[:28], key[28:]
        for i in range(self.round_num):
            g = g[SHIFT[i]:] + g[:SHIFT[i]]
            d = d[SHIFT[i]:] + d[:SHIFT[i]]            
            tmp = g + d
            self.round_keys.append([tmp[x] for x in CP_2])
            
    # Hàm này thực hiện phép thay thế các bit đầu vào bằng cách sử dụng S-box.
    def _substitute(self, R_expand):
        subblocks = [R_expand[k:k+6] for k in range(0, 48, 6)]
        ret = []
        for i in range(8): #For all the sublists
            block = subblocks[i]
            row = int(str(block[0])+str(block[5]), 2)#Get the row with the first and last bit
            column = int(''.join([str(x) for x in block[1:][:-1]]),2) #Column is the 2,3,4,5th bits
            val = S_BOX[i][row][column] #Take the value in the SBOX appropriated for the round (i)
            for j in range(3,-1,-1):
                ret.append((val >> j) & 1)            
        return ret
    
    # Hàm này thực hiện mã hóa DES cho một khối dữ liệu 64 bit.
    def encrypt(self, pt):        
            pt = bytes_to_bits(pt)
            pt = [pt[x] for x in IP]                
            L, R = pt[:32], pt[32:]
            tmp = None
            for i in range(self.round_num):
                R_expand = [R[x] for x in E]
                tmp = xor(self.round_keys[i], R_expand)
                tmp = self._substitute(tmp)
                tmp = [tmp[x] for x in P]
                L, R = R, xor(L, tmp)

            tmp = R + L
            C = [tmp[x] for x in FP]
            return bits_to_bytes(C)
