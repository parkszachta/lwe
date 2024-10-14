import os

def generate_random_data(n):
    result = ""
    for i in range(0, n, 8):
        random_byte = str(bin(int.from_bytes(os.urandom(1))))[2:]
        random_byte = "0" * (8 - len(random_byte)) + random_byte
        cutoff = min(n - i, 8)
        result += random_byte[:cutoff]
    result = list(result)
    result = list(map(int, result))
    return result

q = 2
n = 5
s = ""
num_points_train = 10
num_points_val = 2000
num_points_test = 2000
s = generate_random_data(n)
A = []
B = []
for _ in range(num_points_train):
    a = generate_random_data(n)
    A.append(a)
    b = 0
    for i in range(n):
        b += int(a[i]) * int(s[i])
    b = b % q
    B.append([b])
print(f"s is {s}")
print(f"A is {A}")
print(f"B is {B}")
