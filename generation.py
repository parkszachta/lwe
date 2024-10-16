import random
import sqlite3

# Generates a uniformly random list of n integers in \mathbb{Z}_q
def generate_random_data(n, q):
    result = []
    for _ in range(0, n):
        result.append(random.randrange(q))
    return result

q_input = 3
n_input = 5
num_points_lwe = 10
num_points_uniform = 10
num_points = num_points_lwe + num_points_uniform
s = generate_random_data(n_input, q_input)
A = []
B = []
for i in range(num_points):
    a = generate_random_data(n_input, q_input)
    A.append(a)
    if i < num_points_lwe:
        b = 0
        for i in range(n_input):
            b += int(a[i]) * int(s[i])
        b = b % q_input
    else:
        b = generate_random_data(1, q_input)[0]
    B.append(b)
print(f"s is {s}")
print(f"A is {A}")
print(f"B is {B}")
conn = sqlite3.connect("data.db")
cur = conn.cursor()
s_create_query_component = ""
A_create_query_component = ""
s_insert_query_component = ""
A_insert_query_component = ""
question_marks_component = ""
for i in range(n_input):
    s_create_query_component += f"s_{i + 1} INTEGER, "
    A_create_query_component += f"a_{i + 1} INTEGER, "
    s_insert_query_component += f"s_{i + 1}, "
    A_insert_query_component += f"a_{i + 1}, "
    question_marks_component += "?, ?, "
question_marks_component += "?"
cur.execute(f"DROP TABLE data;")
cur.execute(f"CREATE TABLE IF NOT EXISTS data ({s_create_query_component}{A_create_query_component}b INTEGER);")
s_A_B = []
for i in range(num_points):
    current_point = []
    for item in s:
        current_point.append(item)
    for item in A[i]:
        current_point.append(item)
    current_point.append(B[i])
    s_A_B.append(current_point)
print(f"INSERT INTO data ({s_insert_query_component}{A_insert_query_component}b) VALUES ({question_marks_component});")
print(s_A_B)
cur.executemany(f"INSERT INTO data ({s_insert_query_component}{A_insert_query_component}b) VALUES ({question_marks_component});", s_A_B)
conn.commit()
cur.execute(f"SELECT * FROM data;")
print(cur.fetchall())
conn.commit()
conn.close()