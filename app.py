from flask import Flask, redirect, render_template, request, session
# from flask_session import Session
import numpy as np
import pandas as pd
import io
import pickle
import nnminibatch
from nnminibatch import train, initialize_parameters_deep, L_model_forward
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configure application
app = Flask(__name__)


# # Configure session to use filesystem (instead of signed cookies)
# app.config["SESSION_PERMANENT"] = False
# app.config["SESSION_TYPE"] = "filesystem"
# Session(app)

def read_file(file):
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension == "csv":
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file)
    return data

label_encoder = LabelEncoder()
scaler = StandardScaler()

app.config['TEMPLATES_AUTO_RELOAD'] = True
@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response
X_train= 0 
target = 'att'
label_encoder = LabelEncoder()
scaler = StandardScaler()
@app.route('/', methods =['GET', 'POST'])
def hello():
    global X_train
    global target
    global label_encoder
    global scaler 
    if request.method == 'GET':
        return render_template('testnn.html')
    if request.method == "POST":
        file = request.files.get("file_upload")
        if not file:
            return "Chưa tải file lên", 400
        action = request.form.get("action")
        print(target)
        data_train = read_file(file)

        #Tim cac cot trong data
        if action == "analyze":
            buffer = io.StringIO()
            data_train.info(buf=buffer)
            info_str = buffer.getvalue()
            data_lines = info_str.splitlines()
            print(info_str)
            return render_template("testnn.html" , data_info = data_lines)
        
        # Huan luyen mo hinh
        if action == "train":
            target = request.form.get("target")
            layer_dims = request.form.get("layer_dims")
            choice = request.form.get("choice")
            batch_size = 2 ** int(request.form.get("batch"))
            epochs = int(request.form.get("epochs"))
            learning_rate = 10 ** float(request.form.get("learning_rate"))
            # One hot + tach target
            X_train = data_train.drop(columns=[target, 'label'])
                # X_train = data_train.drop(columns=[target])
            X_train = pd.get_dummies(X_train,drop_first=False)
            
            y_train = label_encoder.fit_transform(data_train[target])
            y_train = y_train.reshape(-1,1)

            # Chuan hoa dl
            
            X_train_scaled = scaler.fit_transform(X_train)

            #tao layer
            layer_dims = list(map(int, layer_dims.split(',')))
            layer_dims.insert(0, X_train_scaled.shape[1])
            parameters = initialize_parameters_deep(layer_dims)
            
            #train
            parameters, _ = train(X_train_scaled,y_train,parameters,batch_size,epochs, learning_rate)

            # Luu trong so ra file
            with open("parameters.pkl", "wb") as f:
                pickle.dump(parameters, f)

            print("Parameters đã được lưu vào file 'parameters.pkl'")

            #Ket qua mo hinh
            AL, _ = L_model_forward(X_train_scaled.T, parameters)
            predicted_labels = np.argmax(AL, axis=0)
            percent_similarity = np.sum(predicted_labels.reshape(-1,1) == y_train) / y_train.shape[0] * 100
            return render_template('testnn.html', daura=f'Train thành công, tỉ lệ dự đoán đúng trên tập dữ liệu {percent_similarity}%')


    
@app.route('/test', methods =['GET', 'POST'])
def test():
    global X_train
    global target
    global label_encoder
    global scaler 
    if request.method == "GET":
        return render_template('testmodel.html')
    if request.method == "POST":
        test_file = request.files.get("file_upload")
        if not test_file:
            return "Chưa tải file lên", 400
        choice = request.form.get("choice")
        
        #Doc file test
        data_test = read_file(test_file)
        if choice == "kiemthu":
            #Xu ly dau vao
            X_test = data_test.drop(columns=['label', target])
            # X_test = test_data.drop(columns=[ target])

            X_test = pd.get_dummies(X_test, drop_first=False)
            print(1)

            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
            X_test_scaled = scaler.transform(X_test)
            y_test = label_encoder.transform(data_test[target])
            y_test = y_test.reshape(-1,1)
            
            #Tinh kha nang du doan tren tap test
            with open("parameters.pkl", "rb") as f:
                parameters = pickle.load(f)
                print("Parameters đã được tải:")
            AL_test, _ = L_model_forward(X_test_scaled.T, parameters)
            predicted_labels_test = np.argmax(AL_test, axis=0)
            percent_similarity_test = np.sum(predicted_labels_test.reshape(-1,1) == y_test) / y_test.shape[0] * 100
            return render_template('testmodel.html', ketqua=f'Tỉ lệ dự đoán đúng trên tập test: {percent_similarity_test}%')
        if choice=='dudoan':

            #Chuan bi du lieu
            X_test = data_test.drop(columns=['label'], errors ='ignore')
            X_test = pd.get_dummies(X_test, drop_first=False)
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
            X_test_scaled = scaler.transform(X_test)


            #Du doan
            AL_test, _ = L_model_forward(X_test_scaled.T, parameters)
            predicted_labels_test = np.argmax(AL_test, axis=0)
            return 0