import numpy as np
class LinRNN:

    def __init__(self, alpha, A, Omega):
        self.alpha = alpha
        self.A = A
        self.Omega = Omega
        self.num_states = alpha.shape[0]
        self.input_dim = A.shape[1]
        self.output_dim = Omega.shape[1] if Omega.ndim > 1 else 1


    def update_dynamics(self, prev, obs):
        #print(prev.shape, self.A.shape)
        # print(prev.shape, self.A.shape)
        next = np.tensordot(prev, self.A, [prev.ndim - 1, 0])
        obs = obs.reshape(len(obs), -1)
        # print(next)
        # print('error', next.shape, obs.shape)
        next = np.tensordot(next, obs, [0, 0])
        # print(next)
        next = next.reshape(-1, )
        return next

    def term_dynamics(self, prev):
        term = np.tensordot(prev, self.Omega, [prev.ndim - 1, 0])
        return term

    def predict(self, obs_sequences):
        current_state = self.alpha
        count = 0
        for o in obs_sequences:
            #print('obs', o.shape)
            # print(count)
            # print(current_state)
            count += 1
            current_state = self.update_dynamics(current_state, o)

        term = self.term_dynamics(current_state).ravel()
        pred = term if self.output_dim > 1 else term[0]
        return np.asarray(pred)


    def build_true_Hankel_tensor(self,l):
        H = self.alpha
        for i in range(l):
            H = np.tensordot(H,self.A,[H.ndim-1,0])
        H = np.tensordot(H,self.Omega,[H.ndim-1,0])
        return H

    def factor_to_classic(self):
        A_tilde = np.sum(self.A, axis = 1)
        A_tilde = np.eye(self.A.shape[0]) - A_tilde
        self.alpha = self.alpha @ A_tilde
        self.Omega = A_tilde @ self.Omega

