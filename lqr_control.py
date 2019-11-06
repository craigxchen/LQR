import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

class control:
    def lqr(A,B,Q,R):
        '''
        Solves for the optimal infinite-horizon, continuous-time LQR controller 
        given linear system (A,B) and cost function parameterized by (Q,R)
        '''
         
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
         
        K = np.matmul(scipy.linalg.inv(R), np.matmul(B.T, S))
         
        eigVals, eigVecs = scipy.linalg.eig(A-np.matmul(B,K))
         
        return K, S, eigVals
    
    def dlqr(A,B,Q,R):
        '''
        Solves for the optimal infinite-horizon, discrete-time LQR controller 
        given linear system (A,B) and cost function parameterized by (Q,R)
        '''
        
        S = scipy.linalg.solve_discrete_are(A, B, Q, R)
    
        F = np.matmul(scipy.linalg.inv(np.matmul(np.matmul(B.T, S), B) + R), (np.matmul(np.matmul(B.T, S), A)))
        
        eigVals, eigVecs = scipy.linalg.eig(A - np.matmul(B, F))
        
        return F, S, eigVals
    
    def simulate_discrete(A,B,K,x0,u0,T):
        '''
        simulates the linear system (A,B) with static control law
        u(t) = K @ x(t)
        from initial condition x0 for T time steps
    
        returns matrices u and x of control and state trajectories, respectively.
        rows are indexed by time
        '''
        x = x0
        u = u0
        for t in range(T):
            u_t = np.matmul(-K, x[:,[-1]])
            x_prime = np.matmul(A, x[:,[-1]]) + np.matmul(B, u_t)
            x = np.hstack((x, x_prime))
            u = np.hstack((u, u_t))
        return x, u
    
    def plot_paths(x1,x2,ylabel,R1,R2):
        fig, ax = plt.subplots()
        colors = [ '#2D328F', '#F15C19' ] # blue, orange
        label_fontsize = 18
    
        t = np.arange(0,x1.shape[0])
        ax.plot(t,x1,color=colors[0],label='R={}'.format(R1[0][0]))
        ax.plot(t,x2,color=colors[1],label='R={}'.format(R2[0][0]))
            
        ax.set_xlabel('Time',fontsize=label_fontsize)
        ax.set_ylabel(ylabel,fontsize=label_fontsize)
        plt.legend(fontsize=label_fontsize)
    
        plt.grid(True)
        plt.show()
        return

