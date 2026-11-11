// Deep Q-Network (DQN) Reinforcement Learning Agent for River Raid
// Pure JavaScript Neural Network + Experience Replay + Target Network

class DenseLayer {
  constructor(inSize, outSize) {
    this.inSize = inSize;
    this.outSize = outSize;
    this.weights = new Float32Array(inSize * outSize);
    this.biases = new Float32Array(outSize);

    // He / Xavier initialization
    const scale = Math.sqrt(2.0 / inSize);
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = (Math.random() * 2 - 1) * scale;
    }
  }

  forward(input, useRelu = true) {
    this.lastInput = input;
    const out = new Float32Array(this.outSize);
    for (let o = 0; o < this.outSize; o++) {
      let sum = this.biases[o];
      const offset = o * this.inSize;
      for (let i = 0; i < this.inSize; i++) {
        sum += this.weights[offset + i] * input[i];
      }
      out[o] = useRelu ? (sum > 0 ? sum : 0.01 * sum) : sum; // LeakyReLU or Linear
    }
    this.lastOutput = out;
    return out;
  }

  backward(gradOutput, lr, useRelu = true) {
    const gradInput = new Float32Array(this.inSize);
    for (let o = 0; o < this.outSize; o++) {
      let dOut = gradOutput[o];
      if (useRelu) {
        dOut *= this.lastOutput[o] > 0 ? 1.0 : 0.01;
      }
      // Clip gradient for stability
      if (dOut > 2.0) dOut = 2.0;
      if (dOut < -2.0) dOut = -2.0;

      const offset = o * this.inSize;
      for (let i = 0; i < this.inSize; i++) {
        gradInput[i] += this.weights[offset + i] * dOut;
        this.weights[offset + i] -= lr * dOut * this.lastInput[i];
      }
      this.biases[o] -= lr * dOut;
    }
    return gradInput;
  }

  copyFrom(other) {
    this.weights.set(other.weights);
    this.biases.set(other.biases);
  }
}

class MLP {
  constructor(inputDim, hiddenDim, outputDim) {
    this.l1 = new DenseLayer(inputDim, hiddenDim);
    this.l2 = new DenseLayer(hiddenDim, hiddenDim);
    this.l3 = new DenseLayer(hiddenDim, outputDim);
  }

  predict(state) {
    const h1 = this.l1.forward(state, true);
    const h2 = this.l2.forward(h1, true);
    return this.l3.forward(h2, false);
  }

  trainStep(state, targetQ, lr) {
    const predQ = this.predict(state);
    const gradOut = new Float32Array(predQ.length);
    for (let i = 0; i < predQ.length; i++) {
      gradOut[i] = predQ[i] - targetQ[i];
    }
    const g2 = this.l3.backward(gradOut, lr, false);
    const g1 = this.l2.backward(g2, lr, true);
    this.l1.backward(g1, lr, true);
  }

  copyFrom(other) {
    this.l1.copyFrom(other.l1);
    this.l2.copyFrom(other.l2);
    this.l3.copyFrom(other.l3);
  }

  toJSON() {
    return {
      l1w: Array.from(this.l1.weights),
      l1b: Array.from(this.l1.biases),
      l2w: Array.from(this.l2.weights),
      l2b: Array.from(this.l2.biases),
      l3w: Array.from(this.l3.weights),
      l3b: Array.from(this.l3.biases)
    };
  }

  fromJSON(data) {
    if (!data) return;
    this.l1.weights.set(data.l1w);
    this.l1.biases.set(data.l1b);
    this.l2.weights.set(data.l2w);
    this.l2.biases.set(data.l2b);
    this.l3.weights.set(data.l3w);
    this.l3.biases.set(data.l3b);
  }
}

// Actions:
// 0: STAY (coast)
// 1: LEFT
// 2: RIGHT
// 3: ACCELERATE (UP)
// 4: BRAKE (DOWN)
const ACTION_NAMES = ['COAST', 'LEFT ←', 'RIGHT →', 'FAST ↑', 'SLOW ↓'];

class RiverRaidDQNAgent {
  constructor() {
    this.stateDim = 13;
    this.actionDim = 5;
    this.hiddenDim = 64;

    this.policyNet = new MLP(this.stateDim, this.hiddenDim, this.actionDim);
    this.targetNet = new MLP(this.stateDim, this.hiddenDim, this.actionDim);
    this.targetNet.copyFrom(this.policyNet);

    // Hyperparameters
    this.gamma = 0.97;
    this.lr = 0.0025;
    this.epsilon = 0.35; // Start with reasonable exploration so users see learning quickly
    this.epsilonMin = 0.02;
    this.epsilonDecay = 0.9992;
    this.batchSize = 32;

    // Replay memory
    this.memoryCapacity = 15000;
    this.memory = [];
    this.memIndex = 0;

    // Telemetry
    this.episode = 1;
    this.totalSteps = 0;
    this.episodeReward = 0;
    this.bestScore = 0;
    this.lastAction = 0;
    this.lastQValues = new Float32Array(this.actionDim);

    this.loadFromStorage();
  }

  act(state, explore = true) {
    const qValues = this.policyNet.predict(state);
    this.lastQValues = qValues;

    if (explore && Math.random() < this.epsilon) {
      this.lastAction = Math.floor(Math.random() * this.actionDim);
      return this.lastAction;
    }

    let bestAct = 0;
    let bestVal = qValues[0];
    for (let a = 1; a < this.actionDim; a++) {
      if (qValues[a] > bestVal) {
        bestVal = qValues[a];
        bestAct = a;
      }
    }
    this.lastAction = bestAct;
    return bestAct;
  }

  remember(state, action, reward, nextState, done) {
    const exp = { state, action, reward, nextState, done };
    if (this.memory.length < this.memoryCapacity) {
      this.memory.push(exp);
    } else {
      this.memory[this.memIndex] = exp;
      this.memIndex = (this.memIndex + 1) % this.memoryCapacity;
    }
    this.episodeReward += reward;
    this.totalSteps++;
  }

  replay() {
    if (this.memory.length < this.batchSize) return;

    for (let b = 0; b < 8; b++) {
      const idx = Math.floor(Math.random() * this.memory.length);
      const { state, action, reward, nextState, done } = this.memory[idx];

      const currentQ = new Float32Array(this.policyNet.predict(state));
      if (done) {
        currentQ[action] = reward;
      } else {
        // Double DQN: online net picks the next action, target net scores it
        const onlineNext = this.policyNet.predict(nextState);
        let nextAct = 0;
        for (let a = 1; a < this.actionDim; a++) {
          if (onlineNext[a] > onlineNext[nextAct]) nextAct = a;
        }
        currentQ[action] = reward + this.gamma * this.targetNet.predict(nextState)[nextAct];
      }

      this.policyNet.trainStep(state, currentQ, this.lr);
    }

    if (this.epsilon > this.epsilonMin) {
      this.epsilon *= this.epsilonDecay;
      if (this.epsilon < this.epsilonMin) this.epsilon = this.epsilonMin;
    }

    if (this.totalSteps % 200 === 0) {
      this.targetNet.copyFrom(this.policyNet);
    }
  }

  onEpisodeEnd(score) {
    if (score > this.bestScore) {
      this.bestScore = score;
    }
    this.episode++;
    this.episodeReward = 0;
    if (this.episode % 5 === 0) {
      this.saveToStorage();
    }
  }

  saveToStorage() {
    try {
      const payload = {
        episode: this.episode,
        epsilon: this.epsilon,
        bestScore: this.bestScore,
        weights: this.policyNet.toJSON()
      };
      localStorage.setItem('riverRaidDQNBrain', JSON.stringify(payload));
    } catch (err) {
      // Ignore storage quota errors
    }
  }

  loadFromStorage() {
    try {
      const raw = localStorage.getItem('riverRaidDQNBrain');
      if (raw) {
        const data = JSON.parse(raw);
        if (data && data.weights) {
          this.policyNet.fromJSON(data.weights);
          this.targetNet.copyFrom(this.policyNet);
          this.episode = data.episode || 1;
          this.epsilon = data.epsilon ?? 0.15;
          this.bestScore = data.bestScore || 0;
        }
      }
    } catch (err) {
      // ignore local storage error
    }

    // Also fetch pre-trained Python DQN brain if served from river_raid.py
    fetch('/api/brain')
      .then((res) => (res.ok ? res.json() : null))
      .then((pyBrain) => {
        if (pyBrain && pyBrain.weights && (pyBrain.bestScore || 0) >= this.bestScore) {
          this.policyNet.fromJSON(pyBrain.weights);
          this.targetNet.copyFrom(this.policyNet);
          this.episode = pyBrain.episode || this.episode;
          this.epsilon = pyBrain.epsilon ?? 0.02;
          this.bestScore = pyBrain.bestScore || this.bestScore;
          this.saveToStorage();
          if (this.onPythonBrainLoaded) this.onPythonBrainLoaded();
        }
      })
      .catch(() => {});
    return true;
  }

  resetBrain() {
    localStorage.removeItem('riverRaidDQNBrain');
    this.policyNet = new MLP(this.stateDim, this.hiddenDim, this.actionDim);
    this.targetNet = new MLP(this.stateDim, this.hiddenDim, this.actionDim);
    this.targetNet.copyFrom(this.policyNet);
    this.memory = [];
    this.memIndex = 0;
    this.episode = 1;
    this.epsilon = 0.45;
    this.episodeReward = 0;
    this.bestScore = 0;
  }
}

window.RiverRaidDQNAgent = RiverRaidDQNAgent;
window.ACTION_NAMES = ACTION_NAMES;
