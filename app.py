import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import sklearn


def make_prediction(data, model):

    with open(model, "rb") as f:
        model = pickle.load(f)

    # print(data, end=" --\n")
    # print(type(data))
    # scale = StandardScaler()
    # X_test = scale.fit_transform(data)

    print(data)

    y_test_pred = model.predict(data)
    y_test_pred = pd.Series(y_test_pred)
    return y_test_pred


def main():
    st.title("Iris Flower Prediction")

    # take input
    sepal_length = st.number_input("Sepal Length")
    sepal_width = st.number_input("Sepal Width")
    petal_length = st.number_input("Petal Length")
    petal_width = st.number_input("Petal Width")

    # sepal_length = 5
    # sepal_width = 3
    # petal_length = 1.6
    # petal_width = 0.2

    print(sepal_length, sepal_width, petal_length, petal_width)

    data = {
        "sepal_length": [sepal_length],
        "sepal_width": [sepal_width],
        "petal_length": [petal_length],
        "petal_width": [petal_width],
    }

    data_df = pd.DataFrame(data)

    print(data_df, end=" ++\n")

    # scale = StandardScaler()
    # X_test = sklearn.preprocessing.StandardScaler().fit_transform(data)

    if st.button("Predict"):
        classes = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
        scale = StandardScaler()
        X = scale.fit_transform(data_df)
        X_test_data = pd.DataFrame(
            X, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
        )

        print(data_df, end=" //////////////////////////\n")
        prediction = make_prediction(X_test_data, "iris_rf.pkl")
        print(prediction, end=" ..........................\n")
        predicted_class_index = prediction[0]
        print(predicted_class_index)
        predicted_class = classes[predicted_class_index]
        st.write(f"Predicted class: {predicted_class}")


if __name__ == "__main__":
    main()
