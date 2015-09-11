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

        public static void Shuffle<T>(this Random rng, IList<T> array)
        {
            int n = array.Count();
            while (n > 1)
            {
                int k = rng.Next(n--);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
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

        public static IList<T[]> Partition<T>(this IList<T> source, int n)
        {
            int length = source.Count();
            IList<T[]> result = new List<T[]>((int)Math.Ceiling(length * 1.0 / n));

            for (int i = 0; i < length; i += n)
            {
                int segmentSize = i + n < length ? n : length - i;
                var segment = new T[segmentSize];

                for (int s = 0; s < segmentSize; s++)
                {
                    segment[s] = source[i + s];
                }
                result.Add(segment);
            }

            return result;
        }
    }
}
