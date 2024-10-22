import numpy as np
import random
import sqlite3
from sklearn.svm import SVC
from sklearn import metrics

q_input = 251
n_input = 30
num_points_train_lwe = 1000
num_points_train_uniform = 1000
num_points_val_lwe = 200
num_points_val_uniform = 200
num_points_test_lwe = 100
num_points_test_uniform = 100
classifications = ["lwe" for _ in range(num_points_train_lwe)] + ["uniform" for _ in range(num_points_train_uniform)]
classifications += ["lwe" for _ in range(num_points_val_lwe)] + ["uniform" for _ in range(num_points_val_uniform)]
classifications += ["lwe" for _ in range(num_points_test_lwe)] + ["uniform" for _ in range(num_points_test_uniform)]
data_split = ["train" for _ in range(num_points_train_lwe + num_points_train_uniform)]
data_split += ["val" for _ in range(num_points_val_lwe + num_points_val_uniform)]
data_split += ["test" for _ in range(num_points_test_lwe + num_points_test_uniform)]
num_points = len(classifications)

# Generates a uniformly random list of n integers in \mathbb{Z}_q
def generate_uniformly_random_list(n, q):
    result = []
    for _ in range(0, n):
        result.append(random.randrange(q))
    return result

# Generates data
def generate_data():
    s = generate_uniformly_random_list(n_input, q_input)
    A = []
    B = []
    for i in range(num_points):
        a = generate_uniformly_random_list(n_input, q_input)
        A.append(a)
        if classifications[i] == "lwe":
            b = 0
            for i in range(n_input):
                b += int(a[i]) * int(s[i]) + round(np.random.normal(loc=0, scale=3))
            b = b % q_input
        else:
            b = generate_uniformly_random_list(1, q_input)[0]
        B.append(b)
    conn = sqlite3.connect("data.db")
    cur = conn.cursor()
    s_create_query_component = ""
    A_create_query_component = ""
    s_insert_query_component = ""
    A_insert_query_component = ""
    question_marks_component = "?, ?, "
    for i in range(n_input):
        s_create_query_component += f"s_{i + 1} INTEGER, "
        A_create_query_component += f"a_{i + 1} INTEGER, "
        s_insert_query_component += f"s_{i + 1}, "
        A_insert_query_component += f"a_{i + 1}, "
        question_marks_component += "?, ?, "
    question_marks_component += "?"
    cur.execute(f"DROP TABLE data;")
    cur.execute(f"CREATE TABLE IF NOT EXISTS data (classification VARCHAR, data_split VARCHAR, {s_create_query_component}{A_create_query_component}b INTEGER);")
    classification_split_s_A_B = []
    for i in range(num_points):
        current_point = []
        current_point.append(classifications[i])
        current_point.append(data_split[i])
        for item in s:
            current_point.append(item)
        for item in A[i]:
            current_point.append(item)
        current_point.append(B[i])
        classification_split_s_A_B.append(current_point)
    cur.executemany(f"INSERT INTO data (classification, data_split, {s_insert_query_component}{A_insert_query_component}b) VALUES ({question_marks_component});", classification_split_s_A_B)
    conn.commit()
    cur.execute(f"SELECT * FROM data;")
    conn.commit()
    conn.close()

def create_X_train_and_y_train():
    conn = sqlite3.connect("data.db")
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM data WHERE data_split = 'train';")
    print()
    X_train_and_y_train = []
    for row in cur.fetchall():
        x_train = list(row[n_input + 2:len(row)])
        y_train = [row[0]]
        X_train_and_y_train.append(x_train + y_train)
    conn.commit()
    random.shuffle(X_train_and_y_train)
    print(X_train_and_y_train)
    X_train = []
    y_train = []
    for row in X_train_and_y_train:
        X_train.append(row[:-1])
        if row[-1] == "lwe":
            y_train.append(1)
        else:
            y_train.append(0)
    clf = SVC()
    clf.fit(X_train, y_train)
    return clf

def create_X_val_and_y_val(clf):
    conn = sqlite3.connect("data.db")
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM data WHERE data_split = 'val';")
    print()
    X_val_and_y_val = []
    for row in cur.fetchall():
        x_val = list(row[n_input + 2:len(row)])
        y_val = [row[0]]
        X_val_and_y_val.append(x_val + y_val)
    conn.commit()
    random.shuffle(X_val_and_y_val)
    print(X_val_and_y_val)
    X_val = []
    y_val = []
    for row in X_val_and_y_val:
        X_val.append(row[:-1])
        if row[-1] == "lwe":
            y_val.append(1)
        else:
            y_val.append(0)
    y_pred = list(clf.predict(X_val))
    print(f"y_val is {y_val}")
    print(f"y_pred is {y_pred}")
    accuracy = metrics.accuracy_score(y_val, y_pred)
    f1_score = metrics.f1_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred)
    recall = metrics.recall_score(y_val, y_pred)
    print(f"Accuracy is {accuracy}")
    print(f"f1_score is {f1_score}")
    print(f"Precision is {precision}")
    print(f"Recall is {recall}")

generate_data()
clf = create_X_train_and_y_train()
create_X_val_and_y_val(clf)