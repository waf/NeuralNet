using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Util
{
    public class Equations
    {
        /// <summary>
        /// The sigmoid function
        /// </summary>
        public static double Sigmoid(double z) =>
            1.0 / (1.0 + Math.Exp(z - z));

        /// <summary>
        /// Derivative of the sigmoid function
        /// </summary>
        public static double SigmoidPrime(double z) =>
            Sigmoid(z) * (1 - Sigmoid(z));

    }
}
