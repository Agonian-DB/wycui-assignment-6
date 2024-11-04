from flask import Flask, render_template, request, url_for
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import base64

app = Flask(__name__)


def generate_plots(N, mu, sigma2, S):
    X = np.random.uniform(0, 1, N)
    Y = X + np.random.normal(mu, sigma2 ** 0.5, N)

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color="blue", label="Data points")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red",
             label=f"Regression line: y = {slope:.2f} * x + {intercept:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Linear Regression with Slope={slope:.2f}, Intercept={intercept:.2f}")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    slopes = []
    intercepts = []

    for _ in range(S):
        X_sim = np.random.uniform(0, 1, N)
        Y_sim = X_sim + np.random.normal(mu, sigma2 ** 0.5, N)
        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    slope_more_extreme = sum(s > slope for s in slopes) / S
    intercept_more_extreme = sum(i < intercept for i in intercepts) / S

    return plot1_path, plot2_path, slope_more_extreme, intercept_more_extreme


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])
        plot1, plot2, slope_extreme, intercept_extreme = generate_plots(N, mu, sigma2, S)
        return render_template("index.html", plot1=plot1, plot2=plot2,
                               slope_extreme=slope_extreme, intercept_extreme=intercept_extreme,
                               N=N, mu=mu, sigma2=sigma2, S=S)

    return render_template("index.html", N=10, mu=0, sigma2=1, S=50)



if __name__ == "__main__":
    app.run(debug=True)
