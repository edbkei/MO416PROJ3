import numpy as np
import matplotlib.pyplot as plt


class CostCalculus:
    @staticmethod
    def h_theta_logistic(params, variables):
        z = params.dot(variables.T)

        return 1.0/(1.0+np.exp(-1.0*z))

    @staticmethod
    def compute_error_logistic(params, x, y):
        m = x.shape[0]
        h = CostCalculus.h_theta_logistic(params, x)
        tmp = y*np.log(h)+(1.0-y)*np.log(1.0-h)

        return np.sum(tmp)/(-1.0*m)

    @staticmethod
    def plot_error_logistic(regressors, classes):
        # Plot error
        fig = plt.figure()
        i = 0
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, 2*len(regressors))]

        for regressor in regressors:
            plt.plot(regressor['train_error'][1:regressor['final_iteration']], color=colors[i],
                     label='Training error ' + classes[i])
            plt.plot(regressor['val_error'][1:regressor['final_iteration']], color=colors[len(regressors)+i],
                     label='Validation error ' + classes[i])
            i += 1
        plt.legend(loc='upper right')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()
        fig.savefig('error_plot_logistic.png')

    @staticmethod
    def h_theta_softmax(params, variables):
        z = params.dot(variables.T)
        z -= np.max(z)
        z_exp = np.exp(z)

        return z_exp/np.sum(z_exp, axis=0)

    @staticmethod
    def compute_error_softmax(params, x, y):
        m = x.shape[0]
        h = CostCalculus.h_theta_softmax(params, x)
        tmp = -np.log(h[y.T == 1])

        return np.sum(tmp)/m

    @staticmethod
    def plot_error_softmax(regressors):
        # Plot error
        fig = plt.figure()
        plt.plot(regressors[0]['train_error'][1:regressors[0]['final_iteration']], '-r', label='Training error')
        plt.plot(regressors[0]['val_error'][1:regressors[0]['final_iteration']], '-b', label='Validation error')
        plt.legend(loc='upper right')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()
        fig.savefig('error_plot_softmax.png')
