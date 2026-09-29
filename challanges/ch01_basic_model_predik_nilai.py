import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def main():
    feature = np.array([1, 2, 3, 4, 5, 6])
    target = np.array([35, 40, 50, 55, 75, 90])

    x = feature.reshape(-1, 1)
    y = target

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=47
    )

    print("X_test:", x_test.ravel())
    print("y_test:", y_test)

    model = LinearRegression()
    model.fit(x_train, y_train)

    print("\nModel sudah belajar.")

    prediction = model.predict([[8]])

    print("Coefficient:", model.coef_)
    print("Intercept:", model.intercept_)
    print("Prediction:", prediction)

if __name__ == '__main__':
    main()