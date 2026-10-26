"""Basic machine learning starter script.

This script uses a tiny synthetic dataset and trains a simple
1D linear regression model using closed-form least squares.
"""

from __future__ import annotations


def train_linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return slope and intercept for y = m*x + b."""
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        raise ValueError("need at least 2 samples")

    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)

    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("cannot fit model when all x values are identical")

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def predict(x: float, slope: float, intercept: float) -> float:
    return slope * x + intercept


def mean_squared_error(xs: list[float], ys: list[float], slope: float, intercept: float) -> float:
    errors = [(predict(x, slope, intercept) - y) ** 2 for x, y in zip(xs, ys)]
    return sum(errors) / len(errors)


def main() -> None:
    # Example toy dataset (hours studied -> exam score)
    x_data = [1, 2, 3, 4, 5, 6, 7, 8]
    y_data = [48, 52, 57, 61, 66, 69, 74, 79]

    split_index = int(len(x_data) * 0.75)
    x_train, x_test = x_data[:split_index], x_data[split_index:]
    y_train, y_test = y_data[:split_index], y_data[split_index:]

    slope, intercept = train_linear_regression(x_train, y_train)
    test_mse = mean_squared_error(x_test, y_test, slope, intercept)

    print("Model: y = m*x + b")
    print(f"m (slope): {slope:.3f}")
    print(f"b (intercept): {intercept:.3f}")
    print(f"Test MSE: {test_mse:.3f}")

    # Try a manual prediction for a new value.
    new_x = 9
    print(f"Prediction for x={new_x}: {predict(new_x, slope, intercept):.2f}")


if __name__ == "__main__":
    main()
