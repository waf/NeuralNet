using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    static class Extensions
    {
        public static double NextGaussian(this Random r, double mu = 0, double sigma = 1)
        {
            // http://stackoverflow.com/questions/218060/random-gaussian-variables

            var u1 = r.NextDouble();
            var u2 = r.NextDouble();

            var rand_std_normal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                Math.Sin(2.0 * Math.PI * u2);

            var rand_normal = mu + sigma * rand_std_normal;

            return rand_normal;
        }

        public static IEnumerable<T> SkipLast<T>(this IEnumerable<T> source, int n = 1)
        {
            var it = source.GetEnumerator();
            bool hasRemainingItems = false;
            var cache = new Queue<T>(n + 1);

            do
            {
                if (hasRemainingItems = it.MoveNext())
                {
                    cache.Enqueue(it.Current);
                    if (cache.Count > n)
                        yield return cache.Dequeue();
                }
            } while (hasRemainingItems);
        }
    }
}
