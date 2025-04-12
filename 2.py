import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        
        self._error_integral = 0
        self._last_error = None
        self._last_output = None
    
    def __call__(self, measurement, dt=1.0):
        error = self.setpoint - measurement
        
        # Proportional term
        p_term = self.Kp * error
        
        # Integral term
        self._error_integral += error * dt
        i_term = self.Ki * self._error_integral
        
        # Derivative term
        if self._last_error is not None:
            d_error = (error - self._last_error) / dt
        else:
            d_error = 0
        d_term = self.Kd * d_error
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits
        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)
        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)
        
        self._last_error = error
        self._last_output = output
        
        return output
    
    def reset(self):
        self._error_integral = 0
        self._last_error = None
        self._last_output = None

# 注意：这个函数应该在类外部定义
def run_cartpole_pid(Kp=10.0, Ki=0.1, Kd=100.0, episodes=10, max_steps=500, render=False):
    env = gym.make('CartPole-v1')
    scores = []
    
    angle_pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=0, output_limits=(-1, 1))
    
    for episode in range(episodes):
        state, _ = env.reset()  # 修改这里
        score = 0
        angle_history = []
        action_history = []
        
        for step in range(max_steps):
            if render:
                env.render()
            
            angle = state[2]  # 角度是状态的第3个元素
            control = angle_pid(angle)
            action = 0 if control < 0 else 1
            
            next_state, reward, terminated, truncated, _ = env.step(action)  # 修改这里
            done = terminated or truncated  # 合并终止条件
            
            angle_history.append(angle)
            action_history.append(action)
            score += reward
            state = next_state
            
            if done:
                break
        
        scores.append(score)
        angle_pid.reset()
        print(f"Episode {episode+1}, Score: {score}")
        
        if render:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(angle_history)
            plt.title('Pendulum Angle')
            plt.xlabel('Step')
            plt.ylabel('Angle (rad)')
            
            plt.subplot(1, 2, 2)
            plt.plot(action_history)
            plt.title('Control Signal')
            plt.xlabel('Step')
            plt.ylabel('Action (0=left, 1=right)')
            plt.yticks([0, 1])
            plt.show()
    
    env.close()
    return scores

# 测试不同PID参数
pid_params = [
    {'Kp': 10.0, 'Ki': 0.1, 'Kd': 100.0},  # 基础参数
    {'Kp': 15.0, 'Ki': 0.05, 'Kd': 120.0}, # 增大Kp和Kd
    {'Kp': 8.0, 'Ki': 0.2, 'Kd': 80.0},   # 减小Kp和Kd
]

all_scores = []
for params in pid_params:
    print(f"\nTesting with Kp={params['Kp']}, Ki={params['Ki']}, Kd={params['Kd']}")
    scores = run_cartpole_pid(**params, episodes=5, render=False)
    all_scores.append(scores)

# 绘制不同参数下的表现
plt.figure(figsize=(10, 6))
for i, (params, scores) in enumerate(zip(pid_params, all_scores)):
    plt.plot(scores, label=f"Kp={params['Kp']}, Ki={params['Ki']}, Kd={params['Kd']}")
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('PID Controller Performance with Different Parameters')
plt.legend()
plt.grid()
plt.savefig('pid_performance_comparison.png')
plt.show()

# 使用最佳参数进行可视化演示
best_params = pid_params[np.argmax([np.mean(scores) for scores in all_scores])]
print(f"\nBest parameters: Kp={best_params['Kp']}, Ki={best_params['Ki']}, Kd={best_params['Kd']}")
run_cartpole_pid(**best_params, episodes=1, max_steps=500, render=True)