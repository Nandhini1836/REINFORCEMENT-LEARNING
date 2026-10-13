/**
 * Real Multi-Layer Perceptron (MLP) for Deep Q-Network
 * Includes He/Kaiming initialization, Forward propagation, ReLU activations,
 * and Adam Optimizer for backpropagation on MSE Loss.
 */

export class NeuralNetwork {
  inputSize: number;
  hidden1Size: number;
  hidden2Size: number;
  outputSize: number;

  // Weights and Biases
  W1: number[][]; // [inputSize x hidden1Size]
  b1: number[];   // [hidden1Size]
  W2: number[][]; // [hidden1Size x hidden2Size]
  b2: number[];   // [hidden2Size]
  W3: number[][]; // [hidden2Size x outputSize]
  b3: number[];   // [outputSize]

  // Adam Optimizer states (First & Second moments)
  mW1: number[][]; vW1: number[][];
  mb1: number[];   vb1: number[];
  mW2: number[][]; vW2: number[][];
  mb2: number[];   vb2: number[];
  mW3: number[][]; vW3: number[][];
  mb3: number[];   vb3: number[];
  t: number;

  constructor(inputSize: number = 14, hidden1Size: number = 48, hidden2Size: number = 48, outputSize: number = 5) {
    this.inputSize = inputSize;
    this.hidden1Size = hidden1Size;
    this.hidden2Size = hidden2Size;
    this.outputSize = outputSize;

    // Initialize weights with He/Kaiming normal initialization
    this.W1 = this.createMatrix(inputSize, hidden1Size, Math.sqrt(2 / inputSize));
    this.b1 = new Array(hidden1Size).fill(0.01);

    this.W2 = this.createMatrix(hidden1Size, hidden2Size, Math.sqrt(2 / hidden1Size));
    this.b2 = new Array(hidden2Size).fill(0.01);

    this.W3 = this.createMatrix(hidden2Size, outputSize, Math.sqrt(2 / hidden2Size));
    this.b3 = new Array(outputSize).fill(0.0);

    // Adam optimizer moments
    this.mW1 = this.createZeroMatrix(inputSize, hidden1Size);
    this.vW1 = this.createZeroMatrix(inputSize, hidden1Size);
    this.mb1 = new Array(hidden1Size).fill(0);
    this.vb1 = new Array(hidden1Size).fill(0);

    this.mW2 = this.createZeroMatrix(hidden1Size, hidden2Size);
    this.vW2 = this.createZeroMatrix(hidden1Size, hidden2Size);
    this.mb2 = new Array(hidden2Size).fill(0);
    this.vb2 = new Array(hidden2Size).fill(0);

    this.mW3 = this.createZeroMatrix(hidden2Size, outputSize);
    this.vW3 = this.createZeroMatrix(hidden2Size, outputSize);
    this.mb3 = new Array(outputSize).fill(0);
    this.vb3 = new Array(outputSize).fill(0);

    this.t = 0;
  }

  private createMatrix(rows: number, cols: number, stdDev: number): number[][] {
    const mat: number[][] = [];
    for (let i = 0; i < rows; i++) {
      const row: number[] = [];
      for (let j = 0; j < cols; j++) {
        // Box-Muller transform for normal distribution
        const u1 = Math.max(1e-7, Math.random());
        const u2 = Math.random();
        const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        row.push(randStdNormal * stdDev);
      }
      mat.push(row);
    }
    return mat;
  }

  private createZeroMatrix(rows: number, cols: number): number[][] {
    return Array.from({ length: rows }, () => new Array(cols).fill(0));
  }

  forward(input: number[]): { qValues: number[]; h1: number[]; h2: number[] } {
    // Hidden Layer 1 (Affine + ReLU)
    const h1: number[] = new Array(this.hidden1Size).fill(0);
    for (let j = 0; j < this.hidden1Size; j++) {
      let sum = this.b1[j];
      for (let i = 0; i < this.inputSize; i++) {
        sum += (input[i] ?? 0) * this.W1[i][j];
      }
      h1[j] = Math.max(0, sum); // ReLU
    }

    // Hidden Layer 2 (Affine + ReLU)
    const h2: number[] = new Array(this.hidden2Size).fill(0);
    for (let j = 0; j < this.hidden2Size; j++) {
      let sum = this.b2[j];
      for (let i = 0; i < this.hidden1Size; i++) {
        sum += h1[i] * this.W2[i][j];
      }
      h2[j] = Math.max(0, sum); // ReLU
    }

    // Output Layer (Linear for Q-values)
    const qValues: number[] = new Array(this.outputSize).fill(0);
    for (let j = 0; j < this.outputSize; j++) {
      let sum = this.b3[j];
      for (let i = 0; i < this.hidden2Size; i++) {
        sum += h2[i] * this.W3[i][j];
      }
      qValues[j] = sum; // Linear output
    }

    return { qValues, h1, h2 };
  }

  /**
   * Train on a batch of experiences using Adam Optimizer on MSE Loss
   */
  trainBatch(
    inputs: number[][],
    targetQValues: number[][],
    learningRate: number = 0.001
  ): number {
    const batchSize = inputs.length;
    if (batchSize === 0) return 0;

    let totalLoss = 0;
    this.t += 1;
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;

    // Gradient accumulators
    const dW1 = this.createZeroMatrix(this.inputSize, this.hidden1Size);
    const db1 = new Array(this.hidden1Size).fill(0);
    const dW2 = this.createZeroMatrix(this.hidden1Size, this.hidden2Size);
    const db2 = new Array(this.hidden2Size).fill(0);
    const dW3 = this.createZeroMatrix(this.hidden2Size, this.outputSize);
    const db3 = new Array(this.outputSize).fill(0);

    for (let b = 0; b < batchSize; b++) {
      const input = inputs[b];
      const target = targetQValues[b];
      const { qValues, h1, h2 } = this.forward(input);

      // Output error: dLoss/dQ = (Q - target)
      const dQ: number[] = new Array(this.outputSize).fill(0);
      for (let j = 0; j < this.outputSize; j++) {
        const diff = qValues[j] - target[j];
        dQ[j] = diff;
        totalLoss += 0.5 * diff * diff;
      }

      // Gradients for Layer 3 (W3, b3)
      const dh2: number[] = new Array(this.hidden2Size).fill(0);
      for (let j = 0; j < this.outputSize; j++) {
        db3[j] += dQ[j] / batchSize;
        for (let i = 0; i < this.hidden2Size; i++) {
          dW3[i][j] += (h2[i] * dQ[j]) / batchSize;
          dh2[i] += this.W3[i][j] * dQ[j];
        }
      }

      // Backprop through ReLU of layer 2
      for (let i = 0; i < this.hidden2Size; i++) {
        if (h2[i] <= 0) dh2[i] = 0;
      }

      // Gradients for Layer 2 (W2, b2)
      const dh1: number[] = new Array(this.hidden1Size).fill(0);
      for (let j = 0; j < this.hidden2Size; j++) {
        db2[j] += dh2[j] / batchSize;
        for (let i = 0; i < this.hidden1Size; i++) {
          dW2[i][j] += (h1[i] * dh2[j]) / batchSize;
          dh1[i] += this.W2[i][j] * dh2[j];
        }
      }

      // Backprop through ReLU of layer 1
      for (let i = 0; i < this.hidden1Size; i++) {
        if (h1[i] <= 0) dh1[i] = 0;
      }

      // Gradients for Layer 1 (W1, b1)
      for (let j = 0; j < this.hidden1Size; j++) {
        db1[j] += dh1[j] / batchSize;
        for (let i = 0; i < this.inputSize; i++) {
          dW1[i][j] += ((input[i] ?? 0) * dh1[j]) / batchSize;
        }
      }
    }

    // Adam parameter updates for W1, b1, W2, b2, W3, b3
    const updateParam = (
      W: number[][],
      dW: number[][],
      mW: number[][],
      vW: number[][]
    ) => {
      for (let i = 0; i < W.length; i++) {
        for (let j = 0; j < W[0].length; j++) {
          mW[i][j] = beta1 * mW[i][j] + (1 - beta1) * dW[i][j];
          vW[i][j] = beta2 * vW[i][j] + (1 - beta2) * dW[i][j] * dW[i][j];
          const mHat = mW[i][j] / (1 - Math.pow(beta1, this.t));
          const vHat = vW[i][j] / (1 - Math.pow(beta2, this.t));
          W[i][j] -= (learningRate * mHat) / (Math.sqrt(vHat) + eps);
        }
      }
    };

    const updateBias = (
      b: number[],
      db: number[],
      mb: number[],
      vb: number[]
    ) => {
      for (let i = 0; i < b.length; i++) {
        mb[i] = beta1 * mb[i] + (1 - beta1) * db[i];
        vb[i] = beta2 * vb[i] + (1 - beta2) * db[i] * db[i];
        const mHat = mb[i] / (1 - Math.pow(beta1, this.t));
        const vHat = vb[i] / (1 - Math.pow(beta2, this.t));
        b[i] -= (learningRate * mHat) / (Math.sqrt(vHat) + eps);
      }
    };

    updateParam(this.W1, dW1, this.mW1, this.vW1);
    updateBias(this.b1, db1, this.mb1, this.vb1);
    updateParam(this.W2, dW2, this.mW2, this.vW2);
    updateBias(this.b2, db2, this.mb2, this.vb2);
    updateParam(this.W3, dW3, this.mW3, this.vW3);
    updateBias(this.b3, db3, this.mb3, this.vb3);

    return totalLoss / batchSize;
  }

  /**
   * Clone weights into this network from another (used for Target Network sync)
   */
  copyFrom(source: NeuralNetwork): void {
    this.W1 = source.W1.map(r => [...r]);
    this.b1 = [...source.b1];
    this.W2 = source.W2.map(r => [...r]);
    this.b2 = [...source.b2];
    this.W3 = source.W3.map(r => [...r]);
    this.b3 = [...source.b3];
  }
}
