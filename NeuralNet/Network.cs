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
        readonly int numLayers;
        readonly int[] layerSizes;
        readonly double[][][] weights;
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
        public void StochasticGradientDescent(IList<Tuple<double[], double[]>> trainingData, int epochs, int miniBatchSize, double eta, IList<Tuple<double[], double[]>> testData)
        {

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
