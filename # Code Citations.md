# Code Citations

## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        a
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0, len(rewards)
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0, len(rewards) - t - 1
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0, len(rewards) - t - 1):
            delta
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0, len(rewards) - t - 1):
            delta = rewards[t+
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0, len(rewards) - t - 1):
            delta = rewards[t+l
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0, len(rewards) - t - 1):
            delta = rewards[t+l] + self.gamma
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0, len(rewards) - t - 1):
            delta = rewards[t+l] + self.gamma*
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0, len(rewards) - t - 1):
            delta = rewards[t+l] + self.gamma*values[t+l
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0, len(rewards) - t - 1):
            delta = rewards[t+l] + self.gamma*values[t+l+1] - values
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0, len(rewards) - t - 1):
            delta = rewards[t+l] + self.gamma*values[t+l+1] - values[t+l]
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0, len(rewards) - t - 1):
            delta = rewards[t+l] + self.gamma*values[t+l+1] - values[t+l]
            ad += ((
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0, len(rewards) - t - 1):
            delta = rewards[t+l] + self.gamma*values[t+l+1] - values[t+l]
            ad += ((self
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0, len(rewards) - t - 1):
            delta = rewards[t+l] + self.gamma*values[t+l+1] - values[t+l]
            ad += ((self.gamma*
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0, len(rewards) - t - 1):
            delta = rewards[t+l] + self.gamma*values[t+l+1] - values[t+l]
            ad += ((self.gamma*self.l
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0, len(rewards) - t - 1):
            delta = rewards[t+l] + self.gamma*values[t+l+1] - values[t+l]
            ad += ((self.gamma*self.lam
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0, len(rewards) - t - 1):
            delta = rewards[t+l] + self.gamma*values[t+l+1] - values[t+l]
            ad += ((self.gamma*self.lam)**
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0, len(rewards) - t - 1):
            delta = rewards[t+l] + self.gamma*values[t+l+1] - values[t+l]
            ad += ((self.gamma*self.lam)**l)
```


## License: MIT
https://github.com/bnelo12/PPO-Implemnetation/blob/62721bbc464b794f249f608d30433973e650c258/PPO.py

```
self, rewards, values):
    advantages = np.zeros_like(rewards)
    for t in range(len(rewards)):
        ad = 0
        for l in range(0, len(rewards) - t - 1):
            delta = rewards[t+l] + self.gamma*values[t+l+1] - values[t+l]
            ad += ((self.gamma*self.lam)**l)*
```

