using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Multi_Treaded_Matrix_Opperations
{
    class Program
    {
        //Declare Stuff
        static Random rng = new Random(); //For Randomization Methods

        static int arraySize = 1000;

        static double[] Array_1 = new double[arraySize];
        static double[] Array_2 = new double[arraySize];
        static double[] Array_3 = new double[arraySize];

        static double[,] Matrix_1 = new double[Array_1.Length, Array_2.Length];

        static DateTime StartTime;
        static DateTime FinishTime;
        static TimeSpan RunTime;

        static void Main(string[] args)
        {
            //Populate Matrix_1, Array_1, and Array_3 with random values
            Matrix_1 = RandomizeMatrix(Matrix_1);
            Array_1 = RandomizeArray(Array_1);
            Array_3 = RandomizeArray(Array_3);

            SingleTreadTest();

            MultiThreadTest();

            Console.ReadLine();
        }

        static void SingleTreadTest()
        {
            //Start Single Treaded Test
            Console.WriteLine("Starting Single Threaded Matrix opperations");

            //Operation 1: Makes Array_2 a weighted sum array of Array_1 and Matrix_1 as weight matrix and applys a sigmoid function on Array_2 
            //To emulate a Feed Forward Method of a layer of a Neural Network
            StartTime = DateTime.Now;

            double sum;
            for (int yy = 0; yy < Array_2.Length; yy++)
            {
                sum = 0;

                for (int xx = 0; xx < Array_1.Length; xx++)
                    sum += Array_1[xx] * Matrix_1[xx, yy];

                Array_2[yy] = Sigmoid(sum);
            }

            FinishTime = DateTime.Now;
            RunTime = FinishTime - StartTime;
            Console.WriteLine("Operation 1 Complete time {0} ms", RunTime.TotalMilliseconds);

            //Operation 2: Takes Array_3 and performs operations to emulate a Back Propergation method of a neural network over Matrix_1
            StartTime = DateTime.Now;

            double[] gradiants = new double[Array_2.Length];
            for (int ii = 0; ii < gradiants.Length; ii++)
                gradiants[ii] = ((1 - Array_2[ii]) * Array_2[ii]) * (Array_3[ii] - Array_2[ii]);

            for (int xx = 0; xx < Array_1.Length; xx++)
                for (int yy = 0; yy < Array_2.Length; yy++)
                    Matrix_1[xx, yy] = 0.03 * gradiants[yy] * Array_1[xx];

            FinishTime = DateTime.Now;
            RunTime = FinishTime - StartTime;
            Console.WriteLine("Operation 2 Complete time {0} ms", RunTime.TotalMilliseconds);
        }

        static void MultiThreadTest()
        {
            //Start Multi Treaded Test
            Console.WriteLine("Starting Multi Threaded Matrix opperations");

            //Multi Threaded version of Operation 1
            StartTime = DateTime.Now;

            double[] sums = new double[Array_2.Length]; 
            Parallel.For(0, Array_2.Length, (yy) =>
            {
                sums[yy] = 0;

                Parallel.For(0, Array_1.Length, (xx) =>
                    {
                        sums[yy] += Array_1[xx] * Matrix_1[xx, yy];
                    });

                Array_2[yy] = Sigmoid(sums[yy]);
            });

            FinishTime = DateTime.Now;
            RunTime = FinishTime - StartTime;
            Console.WriteLine("Operation 1 Complete time {0} ms", RunTime.TotalMilliseconds);

            //Multi Threaded version of Operation 2
            StartTime = DateTime.Now;

            double[] gradiants = new double[Array_2.Length];
            Parallel.For(0, gradiants.Length, (ii) =>
            {
                gradiants[ii] = ((1 - Array_2[ii]) * Array_2[ii]) * (Array_3[ii] - Array_2[ii]);
            });

            Parallel.For(0, Array_1.Length, (xx) =>
            {
                Parallel.For(0, Array_2.Length, (yy) =>
                {
                    Matrix_1[xx, yy] = 0.03 * gradiants[yy] * Array_1[xx];
                });
            });

            FinishTime = DateTime.Now;
            RunTime = FinishTime - StartTime;
            Console.WriteLine("Operation 2 Complete time {0} ms", RunTime.TotalMilliseconds);
        }

        static double[] RandomizeArray(double[] array) 
        {
            for (int ii = 0; ii < array.Length; ii++)
                array[ii] = rng.NextDouble();

            return array;
        }

        static double[,] RandomizeMatrix(double[,] matrix) 
        {
            int width = matrix.GetLength(0);
            int height = matrix.GetLength(1);

            for (int xx = 0; xx < width; xx++)
                for (int yy = 0; yy < height; yy++)
                    matrix[xx, yy] = rng.NextDouble();

            return matrix;
        }

        static double Sigmoid(double x)
        {
            return 1.0 / (1 + Math.Exp(-x));
        }
    }
}
