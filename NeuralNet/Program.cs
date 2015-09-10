using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    class Program
    {
        static void Main(string[] args)
        {
            var nn = new Network(4, 3, 1);
            var result = nn.FeedForward(new double[] { 0.4, 0.6, 0.1, 0.2 });
            return;
        }
    }
}
