import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

def gaussian_kernel(x, theta_1, theta_2):
    return theta_1 * np.exp(- abs(x)**2/theta_2)


def gaussian_regression(x_train, y_train, x_test):
    N = len(y_train)
    # 訓練データに対するカーネル行列の計算
    X = x_train.reshape(-1, 1) - x_train
    K = gaussian_kernel(X, theta_1 = 1, theta_2 = 0.4) + 0.1 * np.identity(len(x_train))
    
    K_inv = np.linalg.inv(K)
    yy = np.dot(K_inv, y_train.reshape(-1, 1))
    mu = np.zeros(len(x_test))
    var = np.zeros(len(x_test))
    for m in range(len(x_test)):
        k = np.zeros(N)
        for n in range(N):
            X = x_train[n] - x_test[m]
            k[n] = gaussian_kernel(X, theta_1 = 1, theta_2 = 0.4)
        s = gaussian_kernel(0, theta_1 = 1, theta_2 = 0.4)
        mu[m] = np.dot(k, yy)
        var[m] = s - np.dot(k.reshape(1, -1), K_inv).dot(k.reshape(1, -1).T)
    
    return mu, var


if __name__ == "__main__":
    x_train = np.array([1.5, 3.6, 4.2, 4.8, 5.7, 7.3, 4.3, 4.34, 4.6, 4.72, 4.5, 4.58])
    y_train = np.array([0.8, 1.75, 1.66, 2.34, 2.12, 1.45, 1.78, 1.92, 2.11, 2.22, 2.01, 2.09])

    x_test = np.arange(1, 8, 0.05)
    y_mu, y_var = gaussian_regression(x_train = x_train, y_train = y_train, x_test = x_test)



    fig, axes = plt.subplots(1, 1, figsize = (10, 8))
    axes.plot(x_train, y_train, "v")
    axes.plot(x_test, y_mu, color = "r")
    axes.fill_between(x_test, y_mu - 2 * np.sqrt(y_var), y_mu + 2 * np.sqrt(y_var), alpha = 0.3)
    plt.show()