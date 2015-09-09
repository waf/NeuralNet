using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Util
{


    public class Functional
    {
        public static IEnumerable<T> Repeat<T>(Func<T> repeater, int times = int.MaxValue)
        {
            for (int i = 0; i < times; i++)
            {
                yield return repeater();
            }
        }
    }
}
