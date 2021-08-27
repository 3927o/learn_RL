import matplotlib
import numpy as np
from matplotlib import pyplot as plt


class BaseRL:
    def __init__(self, k, q_real, epsilon):
        self.k = k
        self.epsilon = epsilon
        self.q = np.zeros(k) + 10
        self.q_real = np.zeros(k) + q_real
        self.statistic = {
            "ave_rewards": [0],
            "is_best_action": [0],
            "best_rate": [0]
        }

    def update_statistic(self, A, R, n):
        self.statistic["is_best_action"].append(A == self.q_real.argmax())
        # self.statistic["ave_rewards"].append((self.statistic["ave_rewards"][-1] * (n-1) + R) / n)
        self.statistic["ave_rewards"].append(self.statistic["ave_rewards"][-1] + 1/n * (R-self.statistic["ave_rewards"][-1]))
        self.statistic["best_rate"].append((self.statistic["best_rate"][-1] * (n-1) + self.statistic["is_best_action"][-1]) / n)

    def update_q_real(self):
        for i in range(self.k):
            self.q_real[i] += np.random.normal(0, 0.01)

    def choose_action(self):
        if np.random.uniform() < self.epsilon or self.q.all() == 0:
            res = np.random.choice(range(self.k))
        else:
            res = self.q.argmax()
        return res

    def update_q(self, A, R):
        raise NotImplementedError

    def train(self, episode):
        for i in range(episode):
            self.n = i
            print(f"executing {i}")
            A = self.choose_action()
            R = self.q_real[A]

            self.update_q(A, R)

            self.update_statistic(A, R, i+1)
            self.update_q_real()


class ARL(BaseRL):
    def __init__(self, k, q_real, epsilon, alpha):
        super(ARL, self).__init__(k, q_real, epsilon)
        self.alpha = alpha

    def update_q(self, A, R):
        self.q[A] += self.alpha * (self.q_real[A] - self.q[A])


class BRL(BaseRL):
    def update_q(self, A, R):
        self.q[A] += 1.0/(self.n+1.0) * (self.q_real[A] - self.q[A])


if __name__ == '__main__':
    episode = 20000

    q_real_a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    q_real_b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    rl_a = ARL(10, q_real_a, 0.1, 0.1)
    rl_b = BRL(10, q_real_b, 0.1)
    rl_a.train(episode)
    rl_b.train(episode)
    print(rl_a.q)
    print(rl_a.q_real)
    print(rl_b.q)
    print(rl_b.q_real)

    plt.xlabel("step")
    plt.ylabel("best action rate",)
    plt.plot(range(2, episode+1), rl_a.statistic["best_rate"][2:], color="green", label="a")
    plt.plot(range(2, episode+1), rl_b.statistic["best_rate"][2:], color="red", label="b")
    plt.show()

    plt.xlabel("step")
    plt.ylabel("ave rewards")
    plt.plot(range(2, episode+1), rl_a.statistic["ave_rewards"][2:], color="green", label="a")
    plt.plot(range(2, episode+1), rl_b.statistic["ave_rewards"][2:], color="red", label="b")
    plt.show()
