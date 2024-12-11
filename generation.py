!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!python rapidsai-csp-utils/colab/pip-install.py

import numpy as np
import random
import sqlite3
from cuml.svm import SVC
from sklearn import metrics
import scipy.stats as st
from google.colab import files

t_input = 50
q_input = 845813581
n_input = 30
h_input = 3
num_points_train_lwe = 250000
num_points_train_uniform = 250000
num_points_val_lwe = 50000
num_points_val_uniform = 50000
num_points_test_lwe = 50000
num_points_test_uniform = 50000
classifications = ["lwe" for _ in range(num_points_train_lwe)] + ["uniform" for _ in range(num_points_train_uniform)]
classifications += ["lwe" for _ in range(num_points_val_lwe)] + ["uniform" for _ in range(num_points_val_uniform)]
classifications += ["lwe" for _ in range(num_points_test_lwe)] + ["uniform" for _ in range(num_points_test_uniform)]
data_split = ["train" for _ in range(num_points_train_lwe + num_points_train_uniform)]
data_split += ["val" for _ in range(num_points_val_lwe + num_points_val_uniform)]
data_split += ["test" for _ in range(num_points_test_lwe + num_points_test_uniform)]
num_points = len(classifications)

# Generates a uniformly random list of n integers such that h of them
# are 1 and n - h of them are 0
def generate_random_binary_message(n, h):
    result = [0 for _ in range(n)]
    one_indices = set()
    while len(one_indices) < h:
        one_indices.add(random.randrange(n))
    for index in one_indices:
        result[index] = 1
    return result

# Generates a uniformly random list of n integers in \mathbb{Z}_q
def generate_uniformly_random_list(n, q):
    result = []
    for _ in range(0, n):
        result.append(random.randrange(q))
    return result

# Generates data
def generate_data(num_trial):
    s = generate_random_binary_message(n_input, h_input)
    print(s)
    A = []
    B = []
    for i in range(num_points):
        a = generate_uniformly_random_list(n_input, q_input)
        A.append(a)
        if classifications[i] == "lwe":
            b = 0
            for i in range(n_input):
                b += int(a[i]) * int(s[i])
            b += round(np.random.normal(loc=0, scale=q_input / 1000))
            b = b % q_input
        else:
            b = generate_uniformly_random_list(1, q_input)[0]
        B.append(b)
    conn = sqlite3.connect(f"data_trial_{num_trial}.db")
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
    cur.execute(f"DROP TABLE IF EXISTS data_trial_{num_trial};")
    cur.execute(f"CREATE TABLE data_trial_{num_trial} (classification VARCHAR, data_split VARCHAR, {s_create_query_component}{A_create_query_component}b INTEGER);")
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
    cur.executemany(f"INSERT INTO data_trial_{num_trial} (classification, data_split, {s_insert_query_component}{A_insert_query_component}b) VALUES ({question_marks_component});", classification_split_s_A_B)
    conn.commit()
    cur.execute(f"SELECT * FROM data_trial_{num_trial};")
    conn.commit()
    conn.close()

def create_X_and_y(record_results, num_trial):
    conn = sqlite3.connect(f"data_trial_{num_trial}.db")
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM data_trial_{num_trial} WHERE data_split = 'train';")
    X_train_and_y_train = []
    for row in cur.fetchall():
        x_train = list(row[n_input + 2:len(row)])
        y_train = [row[0]]
        X_train_and_y_train.append(x_train + y_train)
    conn.commit()
    conn.close()
    X_train = []
    y_train = []
    for row in X_train_and_y_train:
        X_train.append(row[:-1])
        if row[-1] == "lwe":
            y_train.append(1)
        else:
            y_train.append(0)
    X_train_current = np.array(X_train, dtype=np.float32)
    y_train_current = np.array(y_train, dtype=np.float32)
    for C_val_exponent in range(0, 10, 1):
        for gamma_coefficient in [10 ** 0.5, 10, 10 ** 1.5]:
            C_input = 10 ** C_val_exponent
            clf = SVC(C=C_input, gamma = gamma_coefficient/(X_train_current.shape[1] * X_train_current.var()))
            clf.fit(X_train_current, y_train_current)
            create_X_test_and_y_test(clf=clf, record_results=record_results, C=C_input, gamma_coefficient=gamma_coefficient, num_trial=num_trial)

def create_X_test_and_y_test(clf, record_results, C, gamma_coefficient, num_trial):
    conn = sqlite3.connect(f"data_trial_{num_trial}.db")
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM data_trial_{num_trial} WHERE data_split = 'test';")
    print()
    X_test_and_y_test = []
    for row in cur.fetchall():
        x_test = list(row[n_input + 2:len(row)])
        y_test = [row[0]]
        X_test_and_y_test.append(x_test + y_test)
    conn.commit()
    conn.close()
    X_test = []
    y_test = []
    for row in X_test_and_y_test:
        X_test.append(row[:-1])
        if row[-1] == "lwe":
            y_test.append(1)
        else:
            y_test.append(0)
    X_test_current = np.array(X_test, dtype=np.float32)
    y_test_current = np.array(y_test, dtype=np.float32)
    y_pred = list(clf.predict(X_test_current))
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred, zero_division=0)
    precision = metrics.precision_score(y_test, y_pred, zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, zero_division=0)
    if record_results:
        conn_results = sqlite3.connect("results.db")
        cur_results = conn_results.cursor()
        sql_query = "INSERT INTO results (num_trial, h, n, q, c, gamma_coefficient, num_points_train_lwe, num_points_train_uniform, num_points_test_lwe, "
        sql_query += f"num_points_test_uniform, accuracy, f1_score, precision, recall) VALUES ({num_trial}, {h_input}, {n_input}, {q_input}, {C}, {gamma_coefficient}, "
        sql_query += f"{num_points_train_lwe}, {num_points_train_uniform}, {num_points_test_lwe}, {num_points_test_uniform}, "
        sql_query += f"{accuracy}, {f1_score}, {precision}, {recall});"
        cur_results.execute(sql_query)
        conn_results.commit()
        conn_results.close()

def train_run_and_store_results(alphas, num_trial):
    conn_results = sqlite3.connect("results.db")
    cur_results = conn_results.cursor()
    create_X_and_y(record_results=True, num_trial=num_trial)
    for C_val_exponent in range(0, 10, 1):
        for gamma_coefficient in [10 ** 0.5, 10, 10 ** 1.5]:
            C_input = 10 ** C_val_exponent
            cur_results.execute(f"SELECT accuracy FROM results WHERE c = {C_input} AND gamma_coefficient = {gamma_coefficient};")
            accuracies = cur_results.fetchall()
            print(f"When C is {C_input} and gamma_coefficient = {gamma_coefficient}, accuracies is {accuracies}")
            cur_results.execute(f"SELECT f1_score FROM results WHERE c = {C_input} AND gamma_coefficient = {gamma_coefficient};")
            f1_scores = cur_results.fetchall()
            print(f"When C is {C_input} and gamma_coefficient = {gamma_coefficient}, f1_scores is {f1_scores}")
            cur_results.execute(f"SELECT precision FROM results WHERE c = {C_input} AND gamma_coefficient = {gamma_coefficient};")
            precisions = cur_results.fetchall()
            print(f"When C is {C_input} and gamma_coefficient = {gamma_coefficient}, precisions is {precisions}")
            cur_results.execute(f"SELECT recall FROM results WHERE c = {C_input} AND gamma_coefficient = {gamma_coefficient};")
            recalls = cur_results.fetchall()
            print(f"When C is {C_input} and gamma_coefficient = {gamma_coefficient}, recalls is {recalls}")
            for alpha in alphas:
                confidence_interval = st.norm.interval(confidence=alpha, loc=np.mean(accuracies), scale=st.sem(accuracies))
                print(f"When C is {C_input} and gamma_coefficient = {gamma_coefficient}, alpha is {alpha}, accuracies confidence_interval is {confidence_interval}")
                confidence_interval = st.norm.interval(confidence=alpha, loc=np.mean(f1_scores), scale=st.sem(f1_scores))
                print(f"When C is {C_input} and gamma_coefficient = {gamma_coefficient}, alpha is {alpha}, f1_scores confidence_interval is {confidence_interval}")
                confidence_interval = st.norm.interval(confidence=alpha, loc=np.mean(precisions), scale=st.sem(precisions))
                print(f"When C is {C_input} and gamma_coefficient = {gamma_coefficient}, alpha is {alpha}, precisions confidence_interval is {confidence_interval}")
                confidence_interval = st.norm.interval(confidence=alpha, loc=np.mean(recalls), scale=st.sem(recalls))
                print(f"When C is {C_input} and gamma_coefficient = {gamma_coefficient}, alpha is {alpha}, recalls confidence_interval is {confidence_interval}")
    conn_results.commit()
    conn_results.close()

def read_results(alphas):
    conn_results = sqlite3.connect("results.db")
    cur_results = conn_results.cursor()
    for C_val_exponent in range(0, 3, 1):
        for gamma_coefficient in [10 ** 0.5, 10, 10 ** 1.5]:
            C_input = 10 ** C_val_exponent
            cur_results.execute(f"SELECT accuracy FROM results WHERE c = {C_input} AND gamma_coefficient = {gamma_coefficient};")
            accuracies = cur_results.fetchall()
            print(f"When C is {C_input} and gamma_coefficient = {gamma_coefficient}, accuracies is {accuracies}")
            cur_results.execute(f"SELECT f1_score FROM results WHERE c = {C_input} AND gamma_coefficient = {gamma_coefficient};")
            f1_scores = cur_results.fetchall()
            print(f"When C is {C_input} and gamma_coefficient = {gamma_coefficient}, f1_scores is {f1_scores}")
            cur_results.execute(f"SELECT precision FROM results WHERE c = {C_input} AND gamma_coefficient = {gamma_coefficient};")
            precisions = cur_results.fetchall()
            print(f"When C is {C_input} and gamma_coefficient = {gamma_coefficient}, precisions is {precisions}")
            cur_results.execute(f"SELECT recall FROM results WHERE c = {C_input} AND gamma_coefficient = {gamma_coefficient};")
            recalls = cur_results.fetchall()
            print(f"When C is {C_input} and gamma_coefficient = {gamma_coefficient}, recalls is {recalls}")
            for alpha in alphas:
                confidence_interval = st.norm.interval(confidence=alpha, loc=np.mean(accuracies), scale=st.sem(accuracies))
                print(f"When C is {C_input} and gamma_coefficient = {gamma_coefficient}, alpha is {alpha}, accuracies confidence_interval is {confidence_interval}")
                confidence_interval = st.norm.interval(confidence=alpha, loc=np.mean(f1_scores), scale=st.sem(f1_scores))
                print(f"When C is {C_input} and gamma_coefficient = {gamma_coefficient}, alpha is {alpha}, f1_scores confidence_interval is {confidence_interval}")
                confidence_interval = st.norm.interval(confidence=alpha, loc=np.mean(precisions), scale=st.sem(precisions))
                print(f"When C is {C_input} and gamma_coefficient = {gamma_coefficient}, alpha is {alpha}, precisions confidence_interval is {confidence_interval}")
                confidence_interval = st.norm.interval(confidence=alpha, loc=np.mean(recalls), scale=st.sem(recalls))
                print(f"When C is {C_input} and gamma_coefficient = {gamma_coefficient}, alpha is {alpha}, recalls confidence_interval is {confidence_interval}")
    conn_results.commit()
    conn_results.close()

conn_results = sqlite3.connect("results.db")
cur_results = conn_results.cursor()
cur_results.execute("DROP TABLE IF EXISTS results;")
cur_results.execute(f'''CREATE TABLE results (num_trial INTEGER, h INTEGER, n INTEGER, q INTEGER, c FLOAT, gamma_coefficient FLOAT, num_points_train_lwe INTEGER,
                    num_points_train_uniform INTEGER, num_points_test_lwe INTEGER, num_points_test_uniform INTEGER,
                    accuracy FLOAT, f1_score FLOAT, precision FLOAT, recall FLOAT);''')
conn_results.commit()
conn_results.close()
for num_trial in range(t_input):
    generate_data(num_trial=num_trial)
    train_run_and_store_results(alphas=[0.9, 0.95, 0.99], num_trial=num_trial)

read_results(alphas=[0.9, 0.95, 0.99])

files.download('results.db')
for num_trial in range(t_input):
    files.download(f'data_trial_{num_trial}.db')