import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def main():
    x = np.array([
        [40,  1],
        [45,  1],
        [50,  2],
        [55,  2],
        [60,  2],
        [65,  2],
        [70,  2],
        [75,  3],
        [80,  3],
        [85,  3],
        [90,  3],
        [95,  3],
        [100, 3],
        [105, 4],
        [110, 4],
        [115, 4],
        [120, 4],
        [125, 4],
        [130, 4],
        [135, 5],
        [140, 5],
        [145, 5],
        [150, 5],
        [155, 5],
        [160, 5],
        [165, 6],
        [170, 6],
        [175, 6],
        [180, 6],
        [190, 6]
    ])

    y = np.array([
        150,
        165,
        190,
        205,
        220,
        240,
        255,
        275,
        295,
        315,
        330,
        350,
        365,
        390,
        410,
        430,
        450,
        470,
        490,
        515,
        535,
        555,
        580,
        600,
        620,
        650,
        670,
        695,
        720,
        750
    ])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=47
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    print("Model sudah belajar.")

    print("\nPersiapan testing")
    print("data testing", x_test.ravel())

    y_pred = model.predict(x_test)
    print("\nActual", y_test)
    print("Predicted:", y_pred)

    print("\nCoefficient:", model.coef_)
    print("Intercept:", model.intercept_)
    print("Prediction:", y_pred)

if __name__ == '__main__':
    main()