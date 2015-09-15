using NeuralNet.Util;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NeuralNet.Util.Functional;

namespace NeuralNet
{
    public class Network
    {
        double[][] biases;
        double[][][] weights;
        readonly int numLayers;
        readonly int[] layerSizes;
        static readonly Random Random = new Random();

        /// <summary>
        /// The list ``sizes`` contains the number of neurons in the respective layers of the network.
        /// For example, if the list was [2, 3, 1] then it would be a three-layer network, with the first 
        /// layer containing 2 neurons, the second layer 3 neurons, and the third layer 1 neuron.
        /// </summary>
        /// <param name="sizes"></param>
        public Network(params int[] sizes)
        {
            /*
             The biases and weights for the network are initialized randomly, using a Gaussian distribution
             with mean 0, and variance 1.  Note that the first layer is assumed to be an input layer, and 
             by convention we won't set any biases for those neurons, since biases are only ever used in 
             computing the outputs from later layers.
            */
            layerSizes = sizes;
            numLayers = sizes.Length;
            
            biases = GenerateInitialBiases(layerSizes);
            weights = GenerateInitialWeights(layerSizes);
        }

        /// <summary>
        /// Return initial biases for the each neuron connection.
        /// e.g. net.weights[0] is a matrix storing the weights connecting the first and second layers of neurons
        /// </summary>
        /// <param name="sizes">the number of neurons in each layer</param>
        double[][][] GenerateInitialWeights(int[] layerSizes) =>
            // python impl: [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
            layerSizes
                .SkipLast(1)
                .Zip(layerSizes.Skip(1), Tuple.Create)
                .Select(pair => RandomMatrix(pair.Item2, pair.Item1)).ToArray();

        /// <summary>
        /// Return an m by n matrix initialized with random numbers with a Gaussian distribution
        /// </summary>
        double[][] RandomMatrix(int m, int n) =>
            // python impl: [np.random.randn(y, 1) for y in sizes[1:]]
            Repeat(() => Repeat(() => Random.NextGaussian(), n).ToArray(), m).ToArray();

        /// <summary>
        /// Return initial biases for the each neuron.
        /// e.g. given a layerSizes [4,3,1] return array of the form [[r1,r2,r3], [r4]]
        /// </summary>
        /// <param name="sizes">the number of neurons in each layer</param>
        double[][] GenerateInitialBiases(int[] layerSizes) =>
            layerSizes.Skip(1).SelectMany(y => RandomMatrix(1, y)).ToArray();

        /// <summary>
        /// Train the neural network using mini-batch stochastic gradient descent.
        /// The ``trainingData`` is a list of tuples ``(x, y)`` representing the 
        /// training inputs and the desired outputs. 
        /// 
        /// If ``testData`` is provided then the network will be evaluated against
        /// the test data after each epoch, and partial progress printed out. This 
        /// is useful for tracking progress, but slows things down substantially.
        /// </summary>
        public void StochasticGradientDescent(IList<Tuple<double[], double[]>> trainingData, int epochs, int miniBatchSize, double eta, IList<Tuple<double[], double[]>> testData = null)
        {
            for (int j = 0; j < epochs; j++)
            {
                Random.Shuffle(trainingData);
                var miniBatches = trainingData.Partition(miniBatchSize);
                foreach (var miniBatch in miniBatches)
                {
                    this.UpdateMiniBatch(miniBatch, eta);
                }
                if(testData != null)
                {
                    Console.WriteLine("Epoch {0}: {1} / {2}", j, this.Evaluate(testData), testData.Count());
                }
            }
        }

        /// <summary>
        /// Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
        /// </summary>
        /// <param name="miniBatch">the training data</param>
        /// <param name="eta">the learning rate</param>
        private void UpdateMiniBatch(Tuple<double[], double[]>[] miniBatch, double eta)
        {
            var trainingBiases = this.biases.Select(b => new double[b.Length]);
            var trainingWeights = this.weights.Select(w => Matrix.Create(w.Length, w[0].Length));
            foreach (var trainingData in miniBatch)
            {
                Tuple<List<double[]>, List<double[][]>> delta = Backpropagation(trainingData);
                trainingBiases = trainingBiases.Zip(delta.Item1, Matrix.Add);
                trainingWeights = trainingWeights.Zip(delta.Item2, Matrix.Add);
            }
            this.biases = this.biases.Zip(trainingBiases, (b, nb) => Matrix.Add(b, Matrix.Multiply(nb, -eta / miniBatch.Length))).ToArray();
            this.weights = this.weights.Zip(trainingWeights, (w, nw) => Matrix.Add(w, Matrix.Multiply(nw, -eta / miniBatch.Length))).ToArray();
        }

        /// <summary>
        /// Return a tuple representing the gradient for the cost function C_x. 
        /// </summary>
        private Tuple<List<double[]>, List<double[][]>> Backpropagation(Tuple<double[], double[]> trainingData)
        {
            var trainingBiases = this.biases.Select(b => new double[b.Length]).ToArray();
            var trainingWeights = this.weights.Select(w => Matrix.Create(w.Length, w[0].Length)).ToArray();

            // feed forward
            var activation = trainingData.Item1;
            var activations = new List<double[]> { trainingData.Item1 }; //list to store all the activations, layer by layer
            var zs = new List<double[]>(); //list to store all the z vectors, layer by layer
            foreach (var item in this.biases.Zip(this.weights, (bias, weight) => new { bias, weight }))
            {
                var z = Matrix.Add(Matrix.DotProduct(item.weight, activation), item.bias);
                zs.Add(z);
                activation = z.Select(_z => Equations.Sigmoid(_z)).ToArray();
                activations.Add(activation);
            }

            //backward pass
            var y = trainingData.Item2;
            var delta = Matrix.Multiply(
                this.CostDerivative(activations.Last(), y), 
                zs.Last().Select(z => Equations.SigmoidPrime(z)).ToArray());
            trainingBiases[trainingBiases.Length - 1] = delta;
            trainingWeights[trainingWeights.Length - 1] = Matrix.DotProductTransposed(delta, activations[activations.Count() - 2]);

            for(int l = 2; l < this.numLayers; l++)
            {
                var z = zs.Last();
                var sp = z.Select(_z => Equations.SigmoidPrime(_z)).ToArray();
                delta = Matrix.Multiply(Matrix.DotProduct(Matrix.Transpose(this.weights[-l + 1]), delta), sp);
                trainingBiases[trainingBiases.Length - 1] = delta;
                trainingWeights[trainingWeights.Length - 1] = Matrix.DotProductTransposed(delta, activations[-l - 1]);
            }
            return Tuple.Create(trainingBiases.ToList(), trainingWeights.ToList());
        }

        private double[] CostDerivative(double[] outputActivations, double[] y)
        {
            return Matrix.Subtract(outputActivations, y);
        }

        private int Evaluate(IList<Tuple<double[], double[]>> testData)
        {
            Func<double[], int> GetHighestIndex = 
                arr => arr
                        .Select((value, index) => new { Value = value, Index = index })
                        .Aggregate((a, b) => (a.Value > b.Value) ? a : b)
                        .Index;

            return testData
                .Count(kvp => GetHighestIndex(this.FeedForward(kvp.Item1)) == GetHighestIndex(kvp.Item2));
        }

        /// <summary>
        /// Return the output of the network, given ``a`` as input.
        /// </summary>
        public double[] FeedForward(double[] a)
        {
            foreach(var zip in this.biases.Zip(this.weights, (Bias, Weight) => new { Bias, Weight }))
            {
                a = Matrix.Add(Matrix.DotProduct(zip.Weight, a), zip.Bias).Select(entry => Equations.Sigmoid(entry)).ToArray();
            }
            return a;
        }
    }
}
