namespace Emidium.Jnnm.CoreTest
{
    using System;
    using System.CodeDom.Compiler;
    using System.Linq;

    using Emidium.Jnnm.Core;

    using Xunit;

    public class NeuralNetworkMakerTest
    {
        [Fact]
        public void CreateDefinedNeuralNetworkTest()
        {
            var neuralNetworkMaker = new NeuralNetworkMaker();
            var additionNn = neuralNetworkMaker.Poof(this.Add, 2, 1);
            var result = additionNn.Run(new double[2] { 2, 3 });
            var random = new Random();
            double[][] results = new double[1000][];
            for (int i = 0; i < 10; i++)
            {
                var first = random.Next(10);
                var second = random.Next(10);
                var addNnResult = additionNn.Run(new double[] { first, second});
                results[i] = new double[3] { first, second, addNnResult[0] };
            }
            Assert.True(Math.Abs(5 - result[0]) < 0.00001);
        }

        public double[] Add(double[] inputs)
        {
           return new double[] { inputs.Sum() };
        }
    }
}