import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Plot the scree+variance plots for PCA
#Print a table with the propotions and cum for each PCA

def draw_curve_from_diagonal_values(values):
    """
    :param values: singular_values (NOT EIGEN) e.g [17.4, 7.3, 4.3]
    :return:
    """
    squares = np.square(values)
    sum = np.sum(squares)
    rho = squares / (np.ones(len(squares)) * sum)
    df = pd.DataFrame(
        {
            "Singular": values,
            "Singular^2": squares,
            "propotion": rho,
            "var_exp": np.cumsum(rho),
        }
    )
    print(df)

    # Plot variance explained
    threshold = 0.9
    plt.figure()
    plt.plot(range(1, len(rho) + 1), rho, "x-")
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
    plt.plot([1, len(rho)], [threshold, threshold], "k--")
    plt.title("Variance explained by principal components")
    plt.xlabel("Principal component")
    plt.ylabel("Variance explained")
    plt.legend(["Individual", "Cumulative", "Threshold"])
    plt.grid()
    plt.show()

sing_vals=[2.69,2.53,1.05,0.83,0.49,0.31]
draw_curve_from_diagonal_values(sing_vals)
