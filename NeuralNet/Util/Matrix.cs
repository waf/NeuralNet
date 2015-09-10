using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Util
{
    class Matrix
    {
        public static double[] DotProduct(double[][] a, double[] b)
        {
            int aRows = a.Length;
            int aCols = a[0].Length;
            int bRows = b.Length;
            Debug.Assert(aCols == bRows, "Invalid DotProduct dimensions");

            double[] result = new double[aRows];
            for (int i = 0; i < aRows; ++i)
                for (int k = 0; k < aCols; ++k)
                    result[i] += a[i][k] * b[k];

            return result;
        }

        public static double[] Add(double[] a, double[] b)
        {
            int aRows = a.Length;
            int bRows = b.Length;
            Debug.Assert(aRows == bRows, "Invalid Add dimensions");
            double[] result = new double[a.Length];
            for (int i = 0; i < aRows; i++)
            {
                result[i] = a[i] + b[i];
            }
            return result;
        }
    }
}
