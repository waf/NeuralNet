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

        public static double[][] Transpose(double[][] m)
        {
            int length = m[0].Length;
            double[][] result = new double[length][];
            for (int i = 0; i < length; i++)
            {
                result[i] = new double[length];
                for (int j = 0; j < m.Length; j++)
                    result[i][j] = m[j][i];
            }
            return result;
        }

        /// <summary>
        /// Assumes `b` is transposed
        /// </summary>
        public static double[][] DotProductTransposed(double[] a, double[] b)
        {
            int aRows = a.Length;
            int bRows = b.Length;

            double[][] result = new double[aRows][];
            for (int i = 0; i < aRows; ++i)
            {
                result[i] = new double[bRows];
                for (int k = 0; k < bRows; ++k)
                    result[i][k] += a[i] * b[k];

            }

            return result;
        }

        public static double[] Add(double[] a, double[] b)
        {
            int aRows = a.Length;
            int bRows = b.Length;
            Debug.Assert(aRows == bRows, "Invalid Add dimensions");
            double[] result = new double[aRows];
            for (int i = 0; i < aRows; i++)
            {
                result[i] = a[i] + b[i];
            }
            return result;
        }

        public static double[] Subtract(double[] a, double[] b)
        {
            int aRows = a.Length;
            int bRows = b.Length;
            Debug.Assert(aRows == bRows, "Invalid Add dimensions");
            double[] result = new double[aRows];
            for (int i = 0; i < aRows; i++)
            {
                result[i] = a[i] - b[i];
            }
            return result;
        }

        public static double[][] Add(double[][] a, double[][] b)
        {
            int aRows = a.Length;
            int bRows = b.Length;
            Debug.Assert(aRows == bRows, "Invalid Add dimensions");
            double[][] result = new double[aRows][];
            for (int i = 0; i < aRows; i++)
            {
                result[i] = Add(a[i], b[i]);
            }
            return result;
        }

        public static double[] Multiply(double[] a, double b)
        {
            int aRows = a.Length;
            double[] result = new double[aRows];
            for (int i = 0; i < aRows; i++)
            {
                result[i] = a[i] * b;
            }
            return result;
        }


        public static double[] Multiply(double[] a, double[] b)
        {
            int aRows = a.Length;
            double[] result = new double[aRows];
            for (int i = 0; i < aRows; i++)
            {
                result[i] = a[i] * b[i];
            }
            return result;
        }

        public static double[][] Multiply(double[][] a, double b)
        {
            int aRows = a.Length;
            double[][] result = new double[aRows][];
            for (int i = 0; i < aRows; i++)
            {
                result[i] = Multiply(a[i], b);
            }
            return result;
        }

        public static double[][] Create(int m, int n)
        {
            // TODO: investigate actual multidimensional arrays in C# e.g. double[m,n]
            // internet says they're slow. Are they? 
            var matrix = new double[m][];
            for (int i = 0; i < matrix.Length; i++)
            {
                matrix[i] = new double[n];
            }
            return matrix;
        }
    }
}
