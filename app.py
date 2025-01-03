from flask import Flask, redirect, render_template, request,  jsonify
# from flask_session import Session
import numpy as np
import pandas as pd
import io
import pickle
import nnminibatch2
from nnminibatch2 import train, initialize_parameters_deep, L_model_forward, compute_cost
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configure application
app = Flask(__name__)


# # Configure session to use filesystem (instead of signed cookies)
# app.config["SESSION_PERMANENT"] = False
# app.config["SESSION_TYPE"] = "filesystem"
# Session(app)


def percent_similarity(X, y, parameters, activations):
    AL, _ = L_model_forward(X.T, parameters,activations)
    predicted_labels = np.argmax(AL, axis=0)
    percent = np.sum(predicted_labels.reshape(-1,1) == y) / y.shape[0] * 100
    return percent

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
    global scaler
    global label_encoder
    global train_data
    global val_data
    global test_data
    if request.method == "POST":
        if not request.is_json:
            print("check fileasdf")
            files = ["train", "val", "test"]
            uploaded_files = {}
            
            for file_type in files:
        
                file = request.files[file_type]

                filename = file.filename
                if filename == "": 
                    
                    return jsonify({"error": "Khong du file"}), 400
                path = filename.split('.')[-1].lower()
                try:
                    if path == 'csv':
                        uploaded_files[file_type] = pd.read_csv(file)
                    elif path in ['xls', 'xlsx', 'xlsm']:
                        uploaded_files[file_type] = pd.read_excel(file)
                except:
                    
                    # return "File uploaded khong doc duoc", 400
                    return jsonify({"error": "Khong doc duoc file"}), 400
            # columns = uploaded_files['train'].columns.tolist()
            # return jsonify({"columns": columns})
            train_data = uploaded_files['train']
            val_data = uploaded_files['val']
            test_data = uploaded_files['test']
            # Phân tích dữ liệu
            numeric_desc = train_data.describe(include=[float, int]).to_dict()
            categorical_desc = train_data.describe(include=[object, 'category']).to_dict()

            # Xử lý từng cột
            column_info = {}
            for col in train_data.columns:
                null_count = int(train_data[col].isnull().sum())
                if col in numeric_desc:
                    column_info[col] = {
                        "type": "numeric",
                        "describe": numeric_desc[col], # Chứa count, mean, std, min, 25%, 50%, 75%, max
                        "null_count": null_count
                    }
                elif col in categorical_desc:
                    unique_values = train_data[col].dropna().unique().tolist()
                    column_info[col] = {
                        "type": "categorical",
                        "describe": {
                            "count": int( train_data[col].count()),
                            "unique_numbers": str(len(unique_values))
                        },
                        "unique_values": unique_values,
                        "null_count" :null_count
                    
                    }
                    print(int(len(unique_values)))
                else:
                    column_info[col] = {"type": str(train_data[col].dtype)}
            return jsonify({"columns": column_info})
        elif request.is_json:
            print("check train")
            data = request.get_json()  # Nhận dữ liệu JSON từ frontend
            print("Nhận dữ liệu từ frontend:", data)  
            # Truy cập các trường và xử lý chúng
            # 1. Lấy target
            target = data['selectedTarget']

            # 2. Lấy danh sách features
            features = data['selectedFeatures']

            # 3. Lấy danh sách số units và activations từ layers
            layer_dims = [int(layer['units']) for layer in data['layers']]  # Chuyển 'units' thành số nguyên
            activations = [layer['activation'] for layer in data['layers']]  # Lấy giá trị 'activation'

            # 4. Lấy batch_size, chuyển sang 2 mũ kết quả nhận được
            batch_size = 2 ** int(data['batch_size'])

            # 5. Lấy epochs
            epochs = int(data['epoch'])

            # 6. Lấy learning_rate, chuyển thành 10 mũ kết quả nhận được
            activations.append("softmax")
            layer_dims.append(train_data[target].nunique())
            learning_rate = 10 ** float(data['learning_rate'])
            check_target_object = False
            if train_data[target].dtype == 'object':
                check_target_object = True
            # File train
            X_train = train_data[features]
            X_train = pd.get_dummies(X_train,drop_first=False)
            y_train = train_data[target].values
            if check_target_object:
                y_train = label_encoder.fit_transform(train_data[target])
            y_train = y_train.reshape(-1,1)
            X_train_scaled = scaler.fit_transform(X_train)
            train_columns = X_train.columns

            # File val
            X_train_val = val_data[features]
            X_train_val= pd.get_dummies(X_train_val, drop_first=False)
            X_train_val = X_train_val.reindex(columns=train_columns, fill_value=0)
            X_train_val_scaled = scaler.transform(X_train_val)
            y_train_val = val_data[target].values
            if check_target_object:
                y_train_val = label_encoder.transform(val_data[target])
            y_train_val = y_train_val.reshape(-1,1)

            #File test
            X_test = test_data[features]
            X_test= pd.get_dummies(X_test, drop_first=False)
            X_test = X_test.reindex(columns=train_columns, fill_value=0)
            X_test_scaled = scaler.transform(X_test)
            y_test = test_data[target].values
            if check_target_object:
                y_test = label_encoder.transform(test_data[target])
            y_test = y_test.reshape(-1,1)
            #Tạo layer + train
            layer_dims.insert(0, X_train_scaled.shape[1])
            activations.insert(0, None)
            parameters = initialize_parameters_deep(layer_dims)
            parameters, _ = train(X_train_scaled,y_train,parameters,activations ,batch_size,epochs, learning_rate)
            print(target)
            print(features)
            with open('model_data.pkl', 'wb') as file:
                pickle.dump({'parameters': parameters, 'activations': activations, 'target':target,'features':features, 'train_columns': train_columns}, file)
                print("da luu")
            #Ket qua mo hinh file train
            if check_target_object:
                train_result = percent_similarity(X_train_scaled, y_train, parameters, activations)
                # AL, _ = L_model_forward(X_train_scaled.T, parameters,activations)
                # predicted_labels = np.argmax(AL, axis=0)
                # percent_similarity = np.sum(predicted_labels.reshape(-1,1) == y_train) / y_train.shape[0] * 100
                # print(f"Độ chuẩn xác {percent_similarity}")

                #Ket qua mo hinh file val
                val_result = percent_similarity(X_train_val_scaled, y_train_val, parameters, activations)
                # AL, _ = L_model_forward(X_train_val_scaled.T, parameters,activations)
                # predicted_labels = np.argmax(AL, axis=0)
                # percent_similarity = np.sum(predicted_labels.reshape(-1,1) == y_train_val) / y_train_val.shape[0] * 100
                # print(f"Độ chuẩn xác {percent_similarity}")

                #Ket qua mo hinh file test
                test_result = percent_similarity(X_test_scaled, y_test, parameters, activations)
                return jsonify({
                "train": train_result,
                "val": val_result,
                "test": test_result
                            })

            else:
                AL, _ = L_model_forward(X_train_scaled.T, parameters, activations)
                print(f"Train: {compute_cost(AL, y_train,activations[-1])}")

    #Lưu lại model
            
    # Lấy lại thông tin
            # with open('model_data.pkl', 'rb') as file:
            #     data = pickle.load(file)

            # parameters = data['parameters']
            # activations = data['activations']
    return render_template('trainmodel.html')
@app.route('/test', methods =['GET', 'POST'])
def test():
    # Lấy lại thông tin
    with open('model_data.pkl', 'rb') as file:
        data = pickle.load(file)
    parameters = data['parameters']
    activations = data['activations']
    target = data['target']
    features = data['features']
    train_columns = data['train_columns']
    global label_encoder
    global scaler 

    if request.method == "GET":
        return render_template('testmodel.html')
    if request.method == "POST":
        if 'up' not in request.files:
            return jsonify({"error": "Không có file đính kèm."}), 400
        
        file = request.files['up']

        filename = file.filename
        if filename == "":       
            return jsonify({"error": "Khong du file"}), 400
        path = filename.split('.')[-1].lower()
        try:
            if path == 'csv':
                file = pd.read_csv(file)
            elif path in ['xls', 'xlsx', 'xlsm']:
                file = pd.read_excel(file)
                print('asdfwer')
        except:
            return jsonify({"error": "Khong doc duoc file"}), 400
        X = file[features]
        X= pd.get_dummies(X, drop_first=False)
        X = X.reindex(columns=train_columns, fill_value=0)
        X_scaled = scaler.transform(X)
        y = label_encoder.transform(file[target])
        y = y.reshape(-1,1)
        AL, _ = L_model_forward(X_scaled.T, parameters,activations)
        predicted_labels = np.argmax(AL, axis=0)
        predicted_labels = label_encoder.inverse_transform(predicted_labels)
        df = pd.DataFrame(predicted_labels, columns=["Predicted Labels"])


        df.to_excel(f"predicted_labels {target}.xlsx", index=False)

        return jsonify({"message": "đã dự đoạn thành công"}), 200