import numpy as np
import random
import sqlite3
from sklearn.svm import SVC
from sklearn import metrics
import scipy.stats as st

num_trials_per_c_value = 50
q_input = 845813581
n_input = 30
h_input = 15
num_points_train_lwe = 15000#0
num_points_train_uniform = 15000#0
num_points_val_lwe = 5000
num_points_val_uniform = 5000
num_points_test_lwe = 500#0
num_points_test_uniform = 500#0
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
                b += int(a[i]) * int(s[i]) + round(np.random.normal(loc=0, scale=3))
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
    clf_list = []
    for C_val_exponent in range(-5, 6, 1):
        C_input = 10 ** C_val_exponent
        print(f"C is {C_input}")
        clf = SVC(C=C_input)
        clf.fit(X_train, y_train)
        create_X_val_and_y_val(clf=clf, record_results=record_results, C=C_input, num_trial=num_trial)
    return clf_list

def create_X_val_and_y_val(clf, record_results, C, num_trial):
    conn = sqlite3.connect(f"data_trial_{num_trial}.db")
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM data_trial_{num_trial} WHERE data_split = 'val';")
    print()
    X_val_and_y_val = []
    for row in cur.fetchall():
        x_val = list(row[n_input + 2:len(row)])
        y_val = [row[0]]
        X_val_and_y_val.append(x_val + y_val)
    conn.commit()
    conn.close()
    X_val = []
    y_val = []
    for row in X_val_and_y_val:
        X_val.append(row[:-1])
        if row[-1] == "lwe":
            y_val.append(1)
        else:
            y_val.append(0)
    y_pred = list(clf.predict(X_val))
    print("y_val is")
    print(y_val[:100])
    print("y_pred is")
    print(y_pred[:100])
    accuracy = metrics.accuracy_score(y_val, y_pred)
    f1_score = metrics.f1_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred)
    recall = metrics.recall_score(y_val, y_pred)
    print(f"Accuracy is {accuracy}")
    print(f"f1_score is {f1_score}")
    print(f"Precision is {precision}")
    print(f"Recall is {recall}")
    if record_results:
        conn_results = sqlite3.connect("results.db")
        cur_results = conn_results.cursor()
        sql_query = "INSERT INTO results (num_trial, h, n, q, c, num_points_train_lwe, num_points_train_uniform, num_points_val_lwe, "
        sql_query += f"num_points_val_uniform, accuracy, f1_score, precision, recall) VALUES ({num_trial}, {h_input}, {n_input}, {q_input}, {C}, "
        sql_query += f"{num_points_train_lwe}, {num_points_train_uniform}, {num_points_val_lwe}, {num_points_val_uniform}, "
        sql_query += f"{accuracy}, {f1_score}, {precision}, {recall});"
        print(sql_query)
        cur_results.execute(sql_query)
        conn_results.commit()
        conn_results.close()

def train_run_and_store_results(alphas, num_trial):
    conn_results = sqlite3.connect("results.db")
    cur_results = conn_results.cursor()
    create_X_and_y(record_results=True, num_trial=num_trial)
    for C_val_exponent in range(-5, 6, 1):
        C_input = 10 ** C_val_exponent
        cur_results.execute(f"SELECT accuracy FROM results WHERE c = {C_input};")
        accuracies = cur_results.fetchall()
        print(f"When C is {C_input}, accuracies is {accuracies}")
        for alpha in alphas:
            confidence_interval = st.norm.interval(confidence=alpha, loc=np.mean(accuracies), scale=st.sem(accuracies))
            print(f"When C is {C_input}, alpha is {alpha}, confidence_interval is {confidence_interval}")
    conn_results.commit()
    conn_results.close()

def read_results(alphas):
    conn_results = sqlite3.connect("results.db")
    cur_results = conn_results.cursor()
    for C_val_exponent in range(-5, 6, 1):
        C_input = 10 ** C_val_exponent
        cur_results.execute(f"SELECT accuracy FROM results WHERE c = {C_input};")
        accuracies = cur_results.fetchall()
        print(f"When C is {C_input}, accuracies is {accuracies}")
        for alpha in alphas:
            confidence_interval = st.norm.interval(confidence=alpha, loc=np.mean(accuracies), scale=st.sem(accuracies))
            print(f"When C is {C_input}, alpha is {alpha}, confidence_interval is {confidence_interval}")
    conn_results.commit()
    conn_results.close()

# conn_results = sqlite3.connect("results.db")
# cur_results = conn_results.cursor()
# cur_results.execute("DROP TABLE IF EXISTS results;")
# cur_results.execute(f'''CREATE TABLE results (num_trial INTEGER, h INTEGER, n INTEGER, q INTEGER, c FLOAT, num_points_train_lwe INTEGER, 
#             num_points_train_uniform INTEGER, num_points_val_lwe INTEGER, num_points_val_uniform INTEGER, 
#             accuracy FLOAT, f1_score FLOAT, precision FLOAT, recall FLOAT);''')
# conn_results.commit()
# conn_results.close()
# for num_trial in range(num_trials_per_c_value):
#     generate_data(num_trial=num_trial)
#     train_run_and_store_results(alphas=[0.9, 0.95, 0.99], num_trial=num_trial)

read_results(alphas=[0.9, 0.95, 0.99])

# for each of the 50 trials
#    pick s, A, and b values
#    for each C value
#        make the SVM
#        infer with the SVM
#        store the results
#        read the results

# for num_trial in num_trials
#    for c in c_values
#       get the metrics
#       go into the data_trial_{num_trial}
#       analyze the distributions of s, A, and b and how they interrelate to the metrics
