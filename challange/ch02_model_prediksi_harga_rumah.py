import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def main():
    x = np.array([
        [50,  2],
        [60,  2],
        [70,  3],
        [80,  3],
        [90,  3],
        [100, 4],
        [110, 4],
        [120, 4]
    ])

    y = np.array([
        200,
        230,
        290,
        320,
        350,
        410,
        440,
        470
    ])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=47
    )

    print("X_test:", x_test.ravel())
    print("y_test:", y_test)

    model = LinearRegression()
    model.fit(x_train, y_train)

    print("\nModel sudah belajar.")

    prediction = model.predict([[85, 3]])

    print("Coefficient:", model.coef_)
    print("Intercept:", model.intercept_)
    print("Prediction:", prediction)

if __name__ == '__main__':
    main()